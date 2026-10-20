"""CPU checks for the drafter workstream. Run from this directory: python3 -E -P drafter_checks.py
Code under "copied" is transcribed from DeepSpec (DeepSpec/deepspec/modeling/dspark/* at 005e03b),
not imported. Everything labelled 'estimate' is arithmetic, not a measurement."""
import math, torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
GB = 1e9
BW = 936e9  # RTX 3090 peak; achieved is lower, so times below are lower bounds

# ---------------------------------------------------------------- (a) sizes at 27B scale
H, V, NQ, NKV, HD, MLP, NL = 5120, 248320, 32, 8, 128, 17408, 5
attn = H * NQ * HD + 2 * H * NKV * HD + NQ * HD * H
mlp = 3 * H * MLP
backbone = NL * (attn + mlp)
fc = 5 * H * H
print("== (a) parameter counts (Qwen3.8-27B-sized block drafter)")
print(f"per layer attn {attn/1e6:.1f}M mlp {mlp/1e6:.1f}M; 5 layers {backbone/1e9:.3f}B; fc {fc/1e6:.0f}M")
print(f"backbone+fc {(backbone+fc)/1e9:.3f}B vs DFlash2 card 1.92B -> rest {(1.92e9-backbone-fc)/1e6:.0f}M "
      f"(selector codebooks 2*V*r: r=256 -> {2*V*256/1e6:.0f}M, r=384 -> {2*V*384/1e6:.0f}M)")
markov = 2 * V * 256
print(f"DSpark Markov head rank 256: W1 {V*256/1e6:.1f}M + W2 {V*256/1e6:.1f}M = {markov/1e6:.0f}M params; "
      f"confidence head {H+256+1} params")

print("\n== (a) bytes read per decode step (estimate, single stream)")
lm_int4 = 0.65 * GB          # docs: GPTQ-int4 lm_head 0.65 GB (single-user/start_qwen.sh:16)
drafter_w4 = 1.0 * GB        # drafter/quant_dflash2.py:11 '~1.0 GB'
w2_bf16 = V * 256 * 2
draft_slice = lm_int4 * 40960 / V
print(f"DFlash2 W4A16 drafter {drafter_w4/GB:.2f} GB + full int4 lm_head top-k {lm_int4/GB:.2f} GB "
      f"= {(drafter_w4+lm_int4)/GB:.2f} GB -> >= {(drafter_w4+lm_int4)/BW*1e3:.2f} ms")
print(f"DSpark add-on: Markov W2 bf16 {w2_bf16/1e6:.0f} MB read once per sequential position; "
      f"x7 positions = {7*w2_bf16/GB:.2f} GB -> >= {7*w2_bf16/BW*1e3:.2f} ms/step "
      f"(int4 W2: {7*w2_bf16/4/BW*1e3:.2f} ms). DFlash2 selector gathers 16 rows/slot: ~0")
print(f"40,960-row draft slice of int4 lm_head: {draft_slice/GB:.3f} GB, saves {(lm_int4-draft_slice)/GB:.3f} GB "
      f"-> >= {(lm_int4-draft_slice)/BW*1e3:.2f} ms of a 26.5 ms step ({(lm_int4-draft_slice)/BW/26.5e-3*100:.1f}%)")
mtp_per = 0.215 * GB + lm_int4 * 40960 / V
print(f"MTP-4 for scale: (215 MB module + {lm_int4*40960/V/1e6:.0f} MB draft head) x4 sequential = {4*mtp_per/GB:.2f} GB")

# ---------------------------------------------------------------- (a') DeepSpec target cache sizes
print("\n== (a') DeepSpec target-cache bytes (bf16, (n_layers+1)*H*2 per token)")
for name, h, nl in [("Qwen3-4B default", 2560, 5), ("Qwen3.8-27B 5 taps", 5120, 5),
                    ("27B 3 taps", 5120, 3), ("27B 5 taps, no last hidden (CE-only)", 5120, 4)]:
    b = (nl + 1) * h * 2
    print(f"{name:40s} {b:6d} B/token; 7M tok {7e6*b/1e12:.2f} TB; 50M tok {50e6*b/1e12:.1f} TB; "
          f"1.24B tok {1.24e9*b/1e12:.0f} TB")
print(f"top-64 logits (fp16 val + int32 id) instead of last hidden: {64*6} B vs {H*2} B per token")

# ---------------------------------------------------------------- (b) copied DSpark modules, toy size
print("\n== (b) toy DSpark heads + loss (copied from DeepSpec)")


class VanillaMarkov(nn.Module):  # markov_head.py:8-90 (subset)
    def __init__(self, *, vocab_size, markov_rank):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)

    def compute_step_bias(self, token_ids):
        return self.markov_w2(self.markov_w1(token_ids.long()))

    def sample_block_tokens(self, base_logits, first_prev_token_ids):  # greedy branch
        toks, prev = [], first_prev_token_ids.long()
        for i in range(base_logits.shape[1]):
            step = base_logits[:, i] + self.compute_step_bias(prev)
            prev = step.argmax(-1)
            toks.append(prev)
        return torch.stack(toks, 1)


class AcceptRatePredictor(nn.Module):  # common.py:43-49
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, f):
        return self.proj(f).squeeze(-1)


def accept_rate_3d(draft_logits, target_logits):  # loss.py:60-70
    dp, tp = draft_logits.float().softmax(-1), target_logits.float().softmax(-1)
    return (1.0 - 0.5 * (dp - tp).abs().sum(-1)).clamp_(0, 1)


def decay_mask(B, N, K, gamma=4.0):  # loss.py:25-37
    return torch.exp(-torch.arange(K).float() / gamma).view(1, 1, -1).expand(B, N, K)


Vt, Ht, R, K = 512, 64, 16, 7
mh, ch = VanillaMarkov(vocab_size=Vt, markov_rank=R), AcceptRatePredictor(Ht + R)
base = torch.randn(2, K, Vt)
toks = mh.sample_block_tokens(base, torch.tensor([3, 5]))
hid = torch.randn(2, 3, K, Ht)
prev = torch.randint(0, Vt, (2, 3, K))
conf = ch(torch.cat([hid, mh.markov_w1(prev)], -1))
print(f"markov sample shape {tuple(toks.shape)}, confidence logits shape {tuple(conf.shape)}")

# TV identity: speculative-sampling acceptance of x~q against p is sum(min(p,q)) = 1 - TV(p,q)
p = torch.softmax(torch.randn(Vt) * 2, -1)
q = torch.softmax(torch.log(p) + 0.8 * torch.randn(Vt), -1)
n = 400_000
x = torch.multinomial(q, n, replacement=True)
acc = (torch.rand(n) < (p[x] / q[x]).clamp(max=1)).float().mean().item()
tv = accept_rate_3d(torch.log(q)[None], torch.log(p)[None]).item()
print(f"Monte-Carlo acceptance {acc:.4f} vs 1-TV {tv:.4f} vs sum(min) {torch.minimum(p, q).sum():.4f}")
assert abs(acc - tv) < 0.005

# BCE on a soft target is minimised at sigmoid(logit) = target
t = torch.tensor(0.37)
z = torch.zeros((), requires_grad=True)
opt = torch.optim.SGD([z], lr=2.0)
for _ in range(300):
    opt.zero_grad(); F.binary_cross_entropy_with_logits(z, t).backward(); opt.step()
print(f"BCE fixed point sigmoid(z)={z.sigmoid().item():.4f} target {t.item()}")
assert abs(z.sigmoid().item() - 0.37) < 1e-3
w = decay_mask(1, 1, 7)[0, 0]
print("loss position weights exp(-k/4):", [round(v, 3) for v in w.tolist()],
      f"-> position 6 gets {w[6]/w.sum()*100:.1f}% of the weight")

# ---------------------------------------------------------------- (c) acceptance / verify-length sims
print("\n== (c1) draft-vocab slice on DFlash2 (estimate)")


def tau(a_list):
    s, c = 1.0, 1.0
    for a in a_list:
        c *= a; s += c
    return s


def geo_for_tau(target, n):
    lo, hi = 0.0, 0.999
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if tau([m] * n) < target else (lo, m)
    return lo


a7 = geo_for_tau(3.34, 7)  # optimizations.md table: 3.14 / 3.34 tokens per step at k=7
print(f"k=7 geometric per-position acceptance matching tau=3.34: a={a7:.3f}")
for f_oov, a_oov_frac in [(0.025, 0.0), (0.025, 0.5), (0.025, 1.0), (0.01, 1.0)]:
    a_new = a7 - f_oov * a7 * a_oov_frac
    t0, t1 = tau([a7] * 7), tau([a_new] * 7)
    dt = 1 - 0.58 / 26.5
    print(f"OOV share {f_oov:.3f}, OOV tokens currently accepted at {a_oov_frac:.1f}x average: "
          f"tau {t0:.3f}->{t1:.3f} ({(t1/t0-1)*100:+.1f}%), step -2.2% -> net {(t1/t0/dt-1)*100:+.1f}%")

print("\n== (c2) long-context step anatomy from long-context.md:199-207 (step = tok/step / tok/s)")
rows = {"reproduce (dflash2)": (14.19, 154.8), "list cmds (dflash2)": (5.32, 69.8),
        "quote (dflash2)": (2.17, 34.6), "summarize (dflash2)": (2.17, 34.8), "answer (dflash2)": (2.01, 32.1),
        "quote (mtp)": (2.75, 72.9), "summarize (mtp)": (2.68, 71.4), "answer (mtp)": (2.34, 61.9)}
for k, (t, s) in rows.items():
    print(f"  {k:22s} {t/s*1e3:6.1f} ms/step")
s16 = 14.19 / 154.8 * 1e3
s8 = sum(rows[k][0] / rows[k][1] for k in ["quote (dflash2)", "summarize (dflash2)", "answer (dflash2)"]) / 3 * 1e3
c = (s16 - s8) / 8
T0 = s8 - 8 * c
print(f"lookup-hit steps run 16 query tokens, no-hit steps 8 (VLLM_DFLASH2_LOOKUP_ADAPTIVE default on, "
      f"patches/speed-knobs-envs.patch:45) -> c = ({s16:.1f}-{s8:.1f})/8 = {c:.2f} ms per verify token, T0 = {T0:.1f} ms (inferred)")
aqa = geo_for_tau(2.1, 7)
print(f"QA-like: geometric a={aqa:.3f} matches tau 2.1 at 7 drafts")
for L in [8, 6, 5, 4, 3, 2]:
    tt = tau([aqa] * (L - 1)); st = T0 + c * L
    print(f"  fixed verify {L:2d}: tau {tt:.2f}, step {st:.1f} ms -> {tt/st*1e3:.1f} tok/s")
mtp = 2.59 / 37.0e-3
print(f"  MTP CTX=long on the same tasks: ~2.3-2.75 tok/step at ~37-38 ms -> 62-73 tok/s")

print("\n== (c3) per-request truncation with a noisy confidence head, QA-like + reproduce-like mix (estimate)")
g = torch.Generator().manual_seed(1)


def sim(kinds, thr, noise, steps=20000):
    acc_tok, ms = 0.0, 0.0
    for a, n in kinds:
        u = torch.rand(steps, n, generator=g)
        accepted = (u < a).float().cumprod(1).sum(1)
        logit = math.log(a / (1 - a)) + noise * torch.randn(steps, n, generator=g)
        surv = torch.sigmoid(logit).cumprod(1)
        L = (surv >= thr).float().sum(1) if thr > 0 else torch.full((steps,), float(n))
        acc_tok += (torch.minimum(accepted, L) + 1).sum().item()
        ms += (T0 + c * (L + 1)).sum().item()
    return acc_tok / (ms * 1e-3), acc_tok / (steps * len(kinds))


arep = geo_for_tau(14.19, 15)
for name, kinds in [("QA only, 7 drafts", [(aqa, 7)]), ("QA 7 + reproduce 15", [(aqa, 7), (arep, 15)])]:
    b, bt = sim(kinds, 0.0, 0.0)
    out = [f"{name}: fixed {b:.1f} tok/s tau {bt:.2f}"]
    for thr, noise in [(0.3, 0.5), (0.3, 1.5), (0.15, 1.5)]:
        tps, tt = sim(kinds, thr, noise)
        out.append(f"thr {thr} noise {noise}: {tps:.1f} ({(tps/b-1)*100:+.0f}%, tau {tt:.2f})")
    print(" | ".join(out))

print("\n== (c4) chat-like requests (a=0.723, tau 3.34 at 7 drafts) at three slopes (estimate)")


def sim2(kinds, thr, noise, T0x, cx, steps=20000):
    acc_tok, ms = 0.0, 0.0
    for a, n in kinds:
        u = torch.rand(steps, n, generator=g)
        accepted = (u < a).float().cumprod(1).sum(1)
        logit = math.log(a / (1 - a)) + noise * torch.randn(steps, n, generator=g)
        surv = torch.sigmoid(logit).cumprod(1)
        L = (surv >= thr).float().sum(1) if thr > 0 else torch.full((steps,), float(n))
        acc_tok += (torch.minimum(accepted, L) + 1).sum().item()
        ms += (T0x + cx * (L + 1)).sum().item()
    return acc_tok / (ms * 1e-3)


# slope at 25k from the lookup patch comment: 8 extra positions = +27% of a ~26.5 ms step
for label, cx in [("25k: 0.89 ms/token", 0.27 * 26.5 / 8), ("~64k: 2.0 ms/token (interp.)", 2.0), ("112k: %.2f ms/token" % c, c)]:
    T0x = 26.5 - 8 * 0.27 * 26.5 / 8 + 0.0  # drafter+target base at short ctx, ~19.3 ms
    T0x = max(T0x, 1.0) + (0 if cx < 1 else 8 * (cx - 0.89) * 0)  # keep base fixed; slope varies
    b = sim2([(a7, 7)], 0.0, 0.0, T0x, cx)
    r = [f"{label}: fixed 8 {b:.1f} tok/s"]
    thr = cx * b / 1000  # cost-aware: keep a draft while its survival beats tokens-per-ms x its cost
    for noise in [0.0, 0.5, 1.5]:
        r.append(f"cost-aware thr {thr:.2f}, noise {noise}: {(sim2([(a7, 7)], thr, noise, T0x, cx)/b-1)*100:+.1f}%")
    print(" | ".join(r))
print("\nall checks passed")

print("\n== (c5) cost-aware threshold on the long-context mixes of (c3) (estimate)")
for name, kinds in [("QA only, 7 drafts", [(aqa, 7)]), ("QA 7 + reproduce 15", [(aqa, 7), (arep, 15)])]:
    b = sim2(kinds, 0.0, 0.0, T0, c)
    thr = c * b / 1000
    print(f"{name}: fixed {b:.1f} tok/s; cost-aware thr {thr:.2f}: " + ", ".join(
        f"noise {nz} {(sim2(kinds, thr, nz, T0, c)/b-1)*100:+.0f}%" for nz in [0.0, 0.5, 1.5]))
print("\nall checks passed (c5)")
