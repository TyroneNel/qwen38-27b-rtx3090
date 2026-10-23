"""drafter-3: TV-loss repair of the W4A16 DFlash2 drafter (docs/research/deepinfer/drafter.md).

The int4 GPTQ drafter keeps greedy acceptance but loses ~5% at default sampling (3.2 vs 3.4
tokens/step, drafter/README.md): noise in q hurts the acceptance probability, which is
1 - TV(p, q). This script freezes every int4 linear (they stay Marlin-packed, byte-identical)
and trains only the tensors the W4A16 checkpoint already keeps in bf16 -- the grouped convs'
kernel_projection and base_kernel, the candidate selector's codebooks and hidden_projection,
and the norms (~190M params) -- to match the bf16 drafter's selector distribution:

  loss = sum_k w_k TV(q_student(. | prefix_k), q_teacher(. | prefix_k)) / sum_k w_k,
  w_k = exp(-k / gamma), k = 0..6 over the 7 drafted positions (DSpark's gamma = 4).

q is what the rejection sampler sees: softmax(selector edge scores / T) over the drafter's 16
lm_head candidates, truncated to the request's top-k / top-p (VLLM_DFLASH2_DRAFT_TOPK_TOPP,
default on; the model default is T=1, top-k 20, top-p 0.95). The selector walk is made
differentiable by conditioning every position on a fixed predecessor token (--path): the
target's true continuation (default; that is the prefix an accepted draft has) or the
teacher's argmax walk. The candidate top-k is a no-grad selection; the unary logits of the
chosen candidates and the bilinear edge terms carry the gradient. The teacher sees the same
tapped target hidden states, so no target-logit cache is needed; the ceiling is bf16 parity.

Inputs: a drafter-2 capture directory (`capture_dflash2.py --dump-target`; format and reader
in drafter/dflash2_ref.py: meta.json, index.jsonl, taps.bin, ids.bin, ...); the model forward is
drafter/dflash2_ref.py (pure torch). Output: a copy of the W4A16 drafter directory in which
only the trained bf16 tensors differ; vLLM already loads those as bf16, so no loader change.

  python drafter/train_dflash2_tv.py --capture drafter/runs/gate/cap \
      --teacher models/Qwen3.8-27B-DFlash2 --student models/Qwen3.8-27B-DFlash2-W4A16 \
      --target models/Qwen3.8-27B-W4A16-AutoRound-fast --out models/Qwen3.8-27B-DFlash2-W4A16-tv \
      [--max-hours 6] [--epochs 3] [--lr 2e-5] [--anchors 48] [--accum 4] [--path true|teacher]

  python drafter/train_dflash2_tv.py --smoke      # CPU, tiny synthetic drafter, a few steps

Memory on a 24 GB 3090 (bf16): teacher 3.85 GB + student dequantized 3.85 GB + shared target
embed/lm_head ~5.1 GB + fp32 master/Adam state for ~194M params ~2.3 GB + bf16 grads ~0.4 GB
= ~15.5 GB before activations; --anchors bounds the rest (48 anchors x 8 rows, 2k context:
~2-3 GB). Lower --anchors / --max-ctx on OOM.
"""
import argparse, json, math, os, random, shutil, sys, tempfile, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dflash2_ref  # noqa: E402  (drafter-2's pure-torch DFlash2 forward)

DRAFT_LEN = 7               # rows 1..7 of the 8-row block draft anchor_pos+1..+7


def trainable_names(model, which):
    """Checkpoint names of the bf16 tensors to train. `which`: comma list of
    conv, selector, norms, or all."""
    sel = set(which.split(","))
    if "all" in sel:
        sel = {"conv", "selector", "norms"}
    out = []
    for name in model.bf16_state_dict():
        if name in model.quantized_names:
            continue
        kind = ("conv" if ("attention_conv." in name or "mlp_conv." in name) else
                "selector" if name.startswith("candidate_selector.") else
                "norms" if "norm" in name else None)
        if kind in sel:
            out.append(name)
    return out


def truncate_probs(scores, temperature, top_k, top_p):
    """scores [..., K] -> q [..., K]: softmax(scores / T) restricted to the request's top-k /
    top-p (the selector walk kernel's TRUNCATE branch: keep rank < top_k and mass-before <
    top_p), renormalized. The support is chosen without grad; q stays differentiable."""
    logits = scores.float() / temperature
    with torch.no_grad():
        p = torch.softmax(logits, -1)
        order = torch.argsort(logits, -1, descending=True)
        ps = torch.gather(p, -1, order)
        before = torch.cumsum(ps, -1) - ps
        rank = torch.arange(logits.shape[-1], device=logits.device)
        keep_sorted = (rank < top_k) & (before < top_p)
        keep = torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)
    return torch.softmax(logits.masked_fill(~keep, float("-inf")), -1)


def sparse_tv(cand_s, q_s, cand_t, q_t):
    """TV between two distributions supported on 16-token candidate sets each.
    cand_* int [..., K] (unique within a row), q_* [..., K]. Returns [...]."""
    match = (cand_s[..., :, None] == cand_t[..., None, :]).to(q_t.dtype)       # [..., Ks, Kt]
    qt_on_s = (match * q_t[..., None, :]).sum(-1)                               # teacher mass on student ids
    qt_out = (q_t * (1 - match.sum(-2).clamp(max=1))).sum(-1)                   # teacher mass elsewhere
    return 0.5 * ((q_s - qt_on_s).abs().sum(-1) + qt_out)


def anchors_of(rec):
    """Response positions of a capture record with a full 7-token continuation in the span."""
    return list(range(max(rec["prompt_len"], 1), rec["len"] - DRAFT_LEN))


def block_heads(model, ctx_kv, ctx_pos, ids, pos, a):
    """Drafter rows 1..7 for anchors `a` [B] of one sequence: (hidden [B,7,H], candidate ids
    [B,7,K] chosen without grad, unary logits [B,7,K] with grad)."""
    h = model.block_forward(ctx_kv, ctx_pos, ids[a], pos[a])[:, 1:1 + DRAFT_LEN]
    cand, unary = model.candidates(h)
    return h, cand, unary


@torch.no_grad()
def greedy_walk(model, h, cand, unary, anchor_ids):
    """vLLM's selector walk at temperature 0 (speculator._selector_walk_kernel): step 0 scores
    against the anchor, step j against the candidate picked at j-1. Returns tokens [B,7]."""
    full = model.edge_scores_full(cand, unary, h, anchor_ids).float()          # [B,7,K(prev),K]
    prev = torch.zeros(cand.shape[0], dtype=torch.long, device=cand.device)
    toks = []
    for j in range(DRAFT_LEN):
        row = full[torch.arange(cand.shape[0]), j, prev]                        # [B,K]
        prev = torch.nan_to_num(row, nan=-1e30).argmax(-1)
        toks.append(cand[torch.arange(cand.shape[0]), j, prev])
    return torch.stack(toks, 1)


def drafted_q(model, h, cand, unary, prev_ids, args):
    """q at every drafted position given a fixed predecessor per position (prev_ids [B,7],
    prev_ids[:,0] = anchor): the differentiable form of the selector walk."""
    scores = model.edge_scores(cand, unary, h, prev_ids)                        # [B,7,K]
    scores = torch.nan_to_num(scores.float(), nan=-1e30, posinf=1e30, neginf=-1e30)
    return truncate_probs(scores, args.temperature, args.top_k, args.top_p)


def seq_loss(student, teacher, cap, rec, args, device, rng, n_anchors, weights):
    anchors = anchors_of(rec)
    if not anchors:
        return None
    if len(anchors) > n_anchors:
        anchors = sorted(rng.sample(anchors, n_anchors))
    lo = max(0, anchors[0] - args.max_ctx)                    # oldest row any anchor can see
    d = cap.seq(rec, device)                                  # context = rows before each anchor
    ids, pos = d["ids"], d["pos"]
    taps, ctx_pos = d["taps"][lo:anchors[-1]], pos[lo:anchors[-1]]
    del d
    a = torch.as_tensor(anchors, device=device)
    with torch.no_grad():
        t_kv = teacher.context_kv(taps, ctx_pos)
        h_t, cand_t, unary_t = block_heads(teacher, t_kv, ctx_pos, ids, pos, a)
        del t_kv
        if args.path == "true":                               # ids[p .. p+6]: the target's continuation
            prev = ids[a[:, None] + torch.arange(DRAFT_LEN, device=device)]
        else:                                                 # the teacher's own greedy walk
            walk = greedy_walk(teacher, h_t, cand_t, unary_t, ids[a])
            prev = torch.cat([ids[a][:, None], walk[:, :-1]], 1)
        q_t = drafted_q(teacher, h_t, cand_t, unary_t, prev, args)
        del h_t, unary_t
    s_kv = student.context_kv(taps, ctx_pos)
    h_s, cand_s, unary_s = block_heads(student, s_kv, ctx_pos, ids, pos, a)
    q_s = drafted_q(student, h_s, cand_s, unary_s, prev, args)
    tv = sparse_tv(cand_s, q_s, cand_t, q_t)                  # [B,7]
    loss = (tv * weights).sum(-1).mean() / weights.sum()
    return loss, tv.detach()


@torch.no_grad()
def evaluate(student, teacher, cap, recs, args, device, weights):
    """Mean TV(student, teacher) per drafted position and the implied acceptance length
    against the teacher, 1 + sum_k prod_{j<=k} (1 - TV_j) (drafter-2's tau_prob form)."""
    rng = random.Random(1234)
    tot, n = torch.zeros(DRAFT_LEN, device=device), 0
    for rec in recs:
        out = seq_loss(student, teacher, cap, rec, args, device, rng, args.eval_anchors, weights)
        if out is None:
            continue
        tot += out[1].sum(0); n += out[1].shape[0]
    tv = (tot / max(n, 1)).cpu()
    tau = 1 + torch.cumprod(1 - tv, 0).sum().item()
    return tv.tolist(), tau


def save_student(student, names, src_dir, out_dir, log):
    """Copy the W4A16 drafter dir, replacing only the trained bf16 tensors."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    tmp = out_dir.rstrip("/") + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    trained = student.bf16_state_dict()
    with safe_open(os.path.join(src_dir, "model.safetensors"), "pt") as f:
        meta = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    changed = 0
    for k in names:
        assert k in tensors, f"{k} not in {src_dir}/model.safetensors"
        new = trained[k].detach().to(tensors[k].dtype).cpu().contiguous()
        assert new.shape == tensors[k].shape
        tensors[k] = new; changed += 1
    save_file(tensors, os.path.join(tmp, "model.safetensors"), metadata=meta or {"format": "pt"})
    for f in os.listdir(src_dir):
        p = os.path.join(src_dir, f)
        if f != "model.safetensors" and not f.startswith(".") and os.path.isfile(p):
            shutil.copy(p, os.path.join(tmp, f))
    json.dump(log, open(os.path.join(tmp, "tv_repair.json"), "w"), indent=1)
    os.replace(tmp, out_dir)
    print(f"wrote {out_dir}: {changed} bf16 tensors replaced, int4 tensors byte-identical")


def train(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; output dirs are write-once")
    t_start = time.time()
    teacher = dflash2_ref.load_drafter(args.teacher, target_dir=args.target, device=device, dtype=torch.bfloat16)
    student = dflash2_ref.load_drafter(args.student, device=device, dtype=torch.bfloat16, shared=teacher)
    teacher.eval(); student.eval()               # no dropout anywhere; eval() just to be explicit
    for p in list(teacher.parameters()) + list(student.parameters()):
        p.requires_grad_(False)
    names = trainable_names(student, args.train)
    params = dict(student.named_parameters())
    train_p = [params[n] for n in names]
    for p in train_p:
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in train_p)
    print(f"training {len(names)} tensors, {n_train / 1e6:.1f}M params ({args.train}); "
          f"int4 linears frozen: {len(student.quantized_names)}")
    # bf16 params, fp32 master copies + AdamW state (lr ~1e-5 updates underflow bf16)
    master = [p.detach().float().clone().requires_grad_(True) for p in train_p]
    opt = torch.optim.AdamW(master, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    cap = dflash2_ref.Capture(args.capture)
    recs = [r for r in cap.seqs if anchors_of(r)]
    rng = random.Random(args.seed)
    rng.shuffle(recs)
    n_val = max(1, int(len(recs) * args.val_frac))
    val, tr = recs[:n_val], recs[n_val:]
    weights = torch.exp(-torch.arange(DRAFT_LEN, device=device, dtype=torch.float32) / args.gamma)
    steps_per_epoch = max(1, len(tr) // args.accum)
    total = steps_per_epoch * args.epochs if args.max_steps <= 0 else args.max_steps
    sched = lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))

    tv0, tau0 = evaluate(student, teacher, cap, val, args, device, weights)
    print(f"val before: tau vs teacher {tau0:.4f}  TV/pos {[round(x, 4) for x in tv0]}")
    log = {"args": vars(args), "trained": names, "train": [], "val": [{"step": 0, "tau": tau0, "tv": tv0}]}
    best = (tau0, 0, [p.detach().clone() for p in train_p])
    step, t_train = 0, time.time()
    done = False
    for epoch in range(args.epochs):
        rng.shuffle(tr)
        for i0 in range(0, len(tr) - args.accum + 1, args.accum):
            for g in opt.param_groups:
                g["lr"] = args.lr * sched(step)
            acc = 0.0
            for rec in tr[i0:i0 + args.accum]:
                out = seq_loss(student, teacher, cap, rec, args, device, rng, args.anchors, weights)
                if out is None:
                    continue
                (out[0] / args.accum).backward()
                acc += out[0].item() / args.accum
            for m, p in zip(master, train_p):
                m.grad = None if p.grad is None else p.grad.float()
                p.grad = None
            torch.nn.utils.clip_grad_norm_([m for m in master if m.grad is not None], args.clip)
            opt.step(); opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for m, p in zip(master, train_p):
                    p.copy_(m)
            step += 1
            log["train"].append(acc)
            if step % args.log_every == 0 or step == 1:
                el = time.time() - t_train
                print(f"ep {epoch} step {step}/{total} loss {acc:.5f} lr {args.lr * sched(step):.2e} "
                      f"{el / step:.2f}s/step", flush=True)
            hours = (time.time() - t_start) / 3600
            stop = step >= total or hours >= args.max_hours
            if step % args.eval_every == 0 or stop:
                tv, tau = evaluate(student, teacher, cap, val, args, device, weights)
                log["val"].append({"step": step, "tau": tau, "tv": tv, "hours": hours})
                print(f"val step {step}: tau vs teacher {tau:.4f} (start {tau0:.4f})  "
                      f"TV/pos {[round(x, 4) for x in tv]}", flush=True)
                if tau > best[0]:
                    best = (tau, step, [p.detach().clone() for p in train_p])
            if stop:
                done = True
                break
        if done:
            break
    with torch.no_grad():
        for p, b in zip(train_p, best[2]):
            p.copy_(b)
    log["best"] = {"step": best[1], "tau": best[0], "tau_start": tau0}
    print(f"best val tau vs teacher {best[0]:.4f} at step {best[1]} (start {tau0:.4f}); "
          f"{(time.time() - t_start) / 3600:.2f} h")
    if best[1] == 0:
        print("no improvement over the W4A16 drafter; writing it unchanged for the record")
    save_student(student, names, args.student, args.out, log)
    return log


def smoke():
    """CPU smoke on drafter-2's tiny synthetic fixture: a few steps, loss must fall."""
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    root = tempfile.mkdtemp(prefix="dflash2_tv_smoke_")
    fx = dflash2_ref.make_tiny_fixture(root, seed=0, quantize=True, n_seq=12)
    args = parse([
        "--capture", fx["capture"], "--teacher", fx["bf16"], "--student", fx["w4a16"], "--target", fx["target"],
        "--out", os.path.join(root, "out-tv"), "--device", "cpu", "--epochs", "1000",
        "--max-steps", "60", "--accum", "1", "--anchors", "16", "--eval-anchors", "16",
        "--eval-every", "20", "--log-every", "10", "--lr", "3e-4", "--warmup", "5", "--val-frac", "0.2",
    ])
    log = train(args)
    from safetensors.torch import load_file
    src, dst = (load_file(os.path.join(d, "model.safetensors")) for d in (fx["w4a16"], args.out))
    frozen_same = all(torch.equal(src[k], dst[k]) for k in src if k not in log["trained"])
    trained_moved = sum(not torch.equal(src[k], dst[k]) for k in log["trained"])
    print(f"smoke: untrained tensors byte-identical: {frozen_same}; trained tensors changed: "
          f"{trained_moved}/{len(log['trained'])}")
    taus = [v["tau"] for v in log["val"]]
    tl = log["train"]
    first, last = sum(tl[:5]) / 5, sum(tl[-5:]) / 5
    ok = (last < first and taus[-1] > taus[0] and log["best"]["step"] > 0 and frozen_same
          and trained_moved == len(log["trained"]))
    print(f"smoke: train loss first5 {first:.4f} -> last5 {last:.4f}; val tau vs teacher "
          f"{[round(t, 4) for t in taus]} -> {'PASS' if ok else 'FAIL'}")
    shutil.rmtree(root, ignore_errors=True)
    sys.exit(0 if ok else 1)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture"); ap.add_argument("--teacher"); ap.add_argument("--student")
    ap.add_argument("--target"); ap.add_argument("--out"); ap.add_argument("--device", default=None)
    ap.add_argument("--train", default="all", help="comma list: conv, selector, norms, all")
    ap.add_argument("--path", choices=("true", "teacher"), default="true",
                    help="predecessor tokens for the selector walk")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gamma", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = epochs x steps/epoch")
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--accum", type=int, default=4, help="sequences per optimizer step")
    ap.add_argument("--anchors", type=int, default=48, help="anchors sampled per sequence")
    ap.add_argument("--eval-anchors", type=int, default=64)
    ap.add_argument("--max-ctx", type=int, default=2048, help="context rows before the first anchor")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if not a.smoke:
        missing = [k for k in ("capture", "teacher", "student", "target", "out") if getattr(a, k) is None]
        if missing:
            ap.error("missing --" + ", --".join(missing))
    return a


if __name__ == "__main__":
    a = parse()
    smoke() if a.smoke else train(a)
