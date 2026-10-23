"""drafter-2: offline DFlash2 gate. Replays a target capture (capture_dflash2.py
--dump-target) through one or more DFlash2 drafters (bf16, W4A16, a quant-3 requant, a
drafter-3 repair) with the pure-torch forward in dflash2_ref.py, and scores each draft
block against the captured target distribution. No server, no run noise: the same capture
and weights give the same number.

  python drafter/eval_dflash2.py --capture DIR --drafter bf16=models/Qwen3.8-27B-DFlash2 \
      --drafter w4a16=models/Qwen3.8-27B-DFlash2-W4A16 [--target DIR] [--k 7] [--batch 64]
      [--domains a,b] [--modes sample,greedy] [--max-seqs N] [--out result.json]
      [--temperature T --top-k K --top-p P] [--draft-truncate 1|0]
      [--check-known [--known-domain real_think] [--known-labels bf16,w4a16]]
  python drafter/eval_dflash2.py --selftest        # CPU, tiny synthetic drafter/target/capture

Per draft block (anchor x_t at position t, slots 1..k predicting t+1..t+k; the selector's
predecessor at slot j is the reference token x_{t+j-1}, i.e. teacher forcing):
  sample  alpha_j = sum_x min(p_j(x), q_j(x)): p_j the target row t+j-1 after temperature,
          top-k, top-p (from the captured top-64 logits; exact when top_k <= 64), q_j the
          selector walk's proposal softmax(scores/T) over the 16 candidates, truncated to the
          request's top-k/top-p like VLLM_DFLASH2_DRAFT_TOPK_TOPP=1 (--draft-truncate).
          This is the rejection sampler's acceptance probability (DeepSpec accept_rate).
  greedy  alpha_j = 1[candidate argmax == target argmax of row t+j-1]; on a greedy capture
          this is exact: an accepted prefix equals the reference.
Reported per drafter x domain x mode:
  tau_chain   tokens per verify step of the renewal process the server runs: a step at
              anchor t accepts a with P(a=i) = prod_{j<=i} alpha_j (1 - alpha_{i+1}), emits
              a+1 tokens, the next anchor is t+a+1 (expected value, no sampling noise). This
              is the number to hold against vLLM's tokens/step (drafter README protocol).
  tau_anchor  DeepSpec's tau_prob = 1 + sum_j prod_{i<=j} alpha_i averaged over every anchor
              (no post-rejection weighting; higher than tau_chain).
  acc@j       mean alpha_j over anchors (teacher-forced accept rate at position j).
Plus a 95% bootstrap CI over sequences and, for two or more drafters, the paired ratio of
tau_chain against the first drafter with its paired bootstrap CI.

Exit codes (bench/README.md): measurement 0; INVALID 2 (no eligible sequence, capture fails
its alignment check, a --check-known cell is missing or has < 1000 anchors); with
--check-known: 0 PASS, 1 FAIL, 3 when the negative control (target rows shifted by one
position, which must collapse tau) does not drop tau_chain by >= 25%.

--check-known (drafter-2 validation, drafter README "What the measurements said": 8
realistic prompts x 1,024 tokens, k=7): on --known-domain,
  sample  bf16 3.4 +-2%  [3.332, 3.468];  w4a16 3.2 +-2%  [3.136, 3.264]
          gap w4a16/bf16 = 0.941 +-2%  [0.922, 0.960]
  greedy  bf16 3.54 +-2% [3.469, 3.611];  w4a16 3.34-3.65 widened 2%  [3.273, 3.723]
          parity w4a16/bf16 in [0.943, 1.031] widened 2%  [0.925, 1.052]
  repeat  the known cells re-evaluated at half the anchor batch (different GEMM shapes)
          agree within 0.5%
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dflash2_ref as R  # noqa: E402

KNOWN = {  # drafter/README.md:89-90 (8 realistic prompts x 1,024 tokens, fast-variant target)
    ("sample", "bf16"): (3.4 * 0.98, 3.4 * 1.02),
    ("sample", "w4a16"): (3.2 * 0.98, 3.2 * 1.02),
    ("greedy", "bf16"): (3.54 * 0.98, 3.54 * 1.02),
    ("greedy", "w4a16"): (3.34 * 0.98, 3.65 * 1.02),
}
GAP = (3.2 / 3.4 * 0.98, 3.2 / 3.4 * 1.02)
PARITY = (3.34 / 3.54 * 0.98, 3.65 / 3.54 * 1.02)
REPEAT_TOL = 0.005
NEG_DROP = 0.25
MIN_ANCHORS = 1000
ALIGN_MIN = 0.97


# ---------------------------------------------------------------------------------------
# acceptance math


def target_probs(topk_logits, lse, temperature, top_k, top_p):
    """Processed target probabilities over the captured top-K ids (rows sorted descending);
    0 for ids outside the kept support. Exact when 0 < top_k <= K; with top_k disabled the
    mass beyond the captured K is dropped (T=1 uses the true logsumexp for the K kept)."""
    lg = topk_logits.float()
    Kc = lg.shape[-1]
    if 0 < top_k <= Kc:
        x = (lg / temperature).masked_fill(torch.arange(Kc, device=lg.device) >= top_k, float("-inf"))
        p = torch.softmax(x, -1)
    elif temperature == 1.0:
        p = torch.exp(lg - lse[..., None].float())
    else:
        p = torch.softmax(lg / temperature, -1)
    if top_p < 1.0:
        before = torch.cumsum(p, -1) - p
        p = p * (before < top_p)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-30)
    return torch.nan_to_num(p)


def draft_probs(scores, temperature, top_k, top_p, truncate):
    """The selector walk's proposal over its candidates (dflash2/speculator.py
    _selector_walk_kernel with SAMPLE_PROBABILISTIC, TRUNCATE=truncate)."""
    s = scores.float() / temperature
    if truncate:
        pr = torch.softmax(s, -1)
        greater = s[..., None, :] > s[..., :, None]              # [.., i, j] = s_j > s_i
        rank = greater.sum(-1)
        before = (greater * pr[..., None, :]).sum(-1)
        kk = top_k if top_k > 0 else s.shape[-1]
        s = s.masked_fill(~((rank < kk) & (before < top_p)), float("-inf"))
    return torch.softmax(s, -1)


# ---------------------------------------------------------------------------------------
# per-sequence evaluation


@torch.inference_mode()
def eval_sequence(model, cap, s, K, batch, device, sp, truncate, shift=0):
    """alpha [n_anchor, K] (0 where the slot runs past the sequence) for one sequence."""
    d = cap.seq(s, device)
    T, r0 = int(s["len"]), int(s["prompt_len"])
    nA = T - 1 - r0
    if nA < 1:
        return None
    kv = model.context_kv(d["taps"], d["pos"])
    ar = torch.arange(K, device=device)
    out = torch.zeros(nA, K, dtype=torch.float64)
    greedy = s["mode"] == "greedy"
    for a0 in range(r0, T - 1, batch):
        a = torch.arange(a0, min(a0 + batch, T - 1), device=device)
        h = model.block_forward(kv, d["pos"], d["ids"][a], d["pos"][a], k=K)[:, 1:]
        cand, unary = model.candidates(h)
        idx = a[:, None] + ar                                   # slot j predicts idx + 1
        valid = idx <= T - 2
        idxc = idx.clamp(max=T - 2)
        prev = d["ids"][idxc]
        sc = model.edge_scores(cand, unary, h, prev).float()
        rows = (idxc + shift).clamp(max=T - 1)
        if greedy:
            pick = cand.gather(-1, sc.argmax(-1, keepdim=True)).squeeze(-1)
            alpha = (pick == d["topk_ids"][rows, 0]).float()
        else:
            p = target_probs(d["topk_logits"][rows], d["lse"][rows], sp["temperature"], sp["top_k"], sp["top_p"])
            q = draft_probs(sc, sp["temperature"], sp["top_k"], sp["top_p"], truncate)
            eq = cand[..., :, None] == d["topk_ids"][rows][..., None, :]          # [B,K,C,Kc]
            pc = (eq * p[..., None, :]).sum(-1)
            alpha = torch.minimum(pc, q).sum(-1).clamp(max=1.0)
        out[a0 - r0: a0 - r0 + a.shape[0]] = (alpha * valid).double().cpu()
    return out


def chain(alpha):
    """Expected (tokens, steps) of the speculative-decoding renewal process over one
    sequence's anchors (row i = anchor r0+i; n_anchor = tokens left to generate)."""
    nA, K = alpha.shape
    v = np.zeros(nA + K + 2)
    v[0] = 1.0
    toks = steps = 0.0
    for j in range(nA):
        w = v[j]
        if w <= 0:
            continue
        rem = nA - j
        kk = min(K, rem)
        surv = np.concatenate(([1.0], np.cumprod(alpha[j, :kk])))
        P = np.append(surv[:-1] - surv[1:], surv[kk])            # P(a = 0..kk)
        emitted = np.minimum(np.arange(kk + 1) + 1, rem)
        steps += w
        toks += w * float((P * emitted).sum())
        v[j + 1: j + kk + 2] += w * P
    return toks, steps


def summarize_seq(alpha, K):
    toks, steps = chain(alpha.numpy())
    nA = alpha.shape[0]
    full = alpha[: max(0, nA - K + 1)]                            # anchors with all K slots inside
    surv = torch.cumprod(full, 1)
    cnt = torch.tensor([max(0, nA - j) for j in range(K)], dtype=torch.float64)
    return dict(toks=toks, steps=steps, n_anchor=nA, acc_sum=alpha.sum(0).tolist(), acc_cnt=cnt.tolist(),
                tau_anchor_sum=float((1 + surv.sum(1)).sum()), n_full=int(full.shape[0]),
                surv_sum=surv.sum(0).tolist())


def aggregate(rows, K, boot=1000, seed=0):
    toks = np.array([r["toks"] for r in rows])
    steps = np.array([r["steps"] for r in rows])
    acc = np.sum([r["acc_sum"] for r in rows], 0) / np.maximum(np.sum([r["acc_cnt"] for r in rows], 0), 1)
    nf = sum(r["n_full"] for r in rows)
    tau_c = float(toks.sum() / steps.sum())
    rng = np.random.default_rng(seed)
    ci = [float("nan")] * 2
    if len(rows) > 1 and boot:
        idx = rng.integers(0, len(rows), (boot, len(rows)))
        bs = toks[idx].sum(1) / steps[idx].sum(1)
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return dict(n_seq=len(rows), n_anchor=int(sum(r["n_anchor"] for r in rows)), tau_chain=tau_c, tau_chain_ci95=ci,
                tau_anchor=float(sum(r["tau_anchor_sum"] for r in rows) / max(nf, 1)),
                acc_at=[float(x) for x in acc],
                surv_at=[float(x) for x in np.sum([r["surv_sum"] for r in rows], 0) / max(nf, 1)])


def paired(rows_a, rows_b, boot=1000, seed=0):
    ta, sa = np.array([r["toks"] for r in rows_a]), np.array([r["steps"] for r in rows_a])
    tb, sb = np.array([r["toks"] for r in rows_b]), np.array([r["steps"] for r in rows_b])
    ratio = float((tb.sum() / sb.sum()) / (ta.sum() / sa.sum()))
    ci = [float("nan")] * 2
    if len(rows_a) > 1 and boot:
        idx = np.random.default_rng(seed).integers(0, len(rows_a), (boot, len(rows_a)))
        bs = (tb[idx].sum(1) / sb[idx].sum(1)) / (ta[idx].sum(1) / sa[idx].sum(1))
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return dict(ratio=ratio, ci95=ci)


def capture_diagnostics(cap, seqs, sp):
    """Greedy: target top-1 at row r == reference token r+1 (prefill vs decode numerics).
    Sample: reference token inside the target's top-k support."""
    g, s_in = [], []
    for s in seqs:
        a, n, r0 = s["off"], s["len"], s["prompt_len"]
        ref = np.asarray(cap.ids[a + r0: a + n])
        top = np.asarray(cap.topk_ids[a + r0 - 1: a + n - 1])
        if s["mode"] == "greedy":
            g.append(float((top[:, 0] == ref).mean()))
        elif sp["top_k"] > 0:
            s_in.append(float((top[:, : sp["top_k"]] == ref[:, None]).any(1).mean()))
    return dict(greedy_top1_agree=float(np.mean(g)) if g else None,
                sample_ref_in_topk=float(np.mean(s_in)) if s_in else None)


# ---------------------------------------------------------------------------------------
# driver


def run(args):
    cap = R.Capture(args.capture)
    sp = dict(cap.meta.get("sampling") or {})
    sp = dict(temperature=float(sp.get("temperature") or 1.0), top_k=int(sp.get("top_k") or 0),
              top_p=float(sp.get("top_p") or 1.0))
    for k in ("temperature", "top_k", "top_p"):
        if getattr(args, k) is not None:
            sp[k] = type(sp[k])(getattr(args, k))
    if sp["temperature"] <= 0:
        raise SystemExit("--temperature must be > 0 (greedy sequences are scored greedily anyway)")
    if (cap.meta.get("sampling") or {}).get("min_p"):
        print(f"WARNING: capture sampling has min_p={cap.meta['sampling']['min_p']}; not modelled", flush=True)
    seqs = [s for s in cap.seqs
            if (not args.domains or s["domain"] in args.domains.split(","))
            and (not args.modes or s["mode"] in args.modes.split(","))]
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    labels = [d.split("=", 1)[0] for d in args.drafter]
    paths = [d.split("=", 1)[1] for d in args.drafter]
    target = args.target or cap.meta.get("target")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    res = dict(capture=os.path.abspath(args.capture), target=target, k=args.k, sampling=sp,
               draft_truncate=bool(args.draft_truncate), drafters=dict(zip(labels, paths)), device=device,
               results={}, paired={}, per_seq={})
    if not seqs:
        print("INVALID: no sequence matches --domains/--modes", flush=True)
        return res, 2
    diag = capture_diagnostics(cap, seqs, sp)
    res["diagnostics"] = diag
    print(f"{len(seqs)} sequences, {sum(s['len'] - 1 - s['prompt_len'] for s in seqs)} anchors, k={args.k}, "
          f"sampling {sp}, draft truncate {bool(args.draft_truncate)}, device {device}; diagnostics {diag}",
          flush=True)
    invalid = []
    if diag["greedy_top1_agree"] is not None and diag["greedy_top1_agree"] < ALIGN_MIN:
        invalid.append(f"greedy top-1 vs reference agreement {diag['greedy_top1_agree']:.4f} < {ALIGN_MIN}")
    if diag["sample_ref_in_topk"] is not None and diag["sample_ref_in_topk"] < ALIGN_MIN:
        invalid.append(f"sampled reference inside target top-{sp['top_k']} {diag['sample_ref_in_topk']:.4f} < {ALIGN_MIN}")

    per = {}
    shared = None
    extra = {}
    known_seqs = [s for s in seqs if s["domain"] == args.known_domain] if args.check_known else []
    for li, (label, path) in enumerate(zip(labels, paths)):
        t0 = time.time()
        m = R.load_drafter(path, target, device=device, dtype=torch.bfloat16, shared=shared)
        shared = SimpleNamespace(embed=m.embed, lm_head=m.lm_head)
        print(f"[{label}] {path}: {len(m.quantized_names)} packed modules dequantized, "
              f"load {time.time() - t0:.0f}s", flush=True)
        rows = []
        for i, s in enumerate(seqs):
            al = eval_sequence(m, cap, s, args.k, args.batch, device, sp, args.draft_truncate)
            if al is None:
                continue
            rows.append(dict(summarize_seq(al, args.k), seq=s["seq"], domain=s["domain"], mode=s["mode"]))
            if (i + 1) % 25 == 0:
                print(f"  [{label}] {i + 1}/{len(seqs)} sequences, {time.time() - t0:.0f}s", flush=True)
        per[label] = rows
        print(f"  [{label}] done in {time.time() - t0:.0f}s", flush=True)
        if args.check_known:
            ks = [s for s in known_seqs]
            rep = []
            for s in ks:
                al = eval_sequence(m, cap, s, args.k, max(1, args.batch // 2), device, sp, args.draft_truncate)
                if al is not None:
                    rep.append(dict(summarize_seq(al, args.k), seq=s["seq"], domain=s["domain"], mode=s["mode"]))
            extra[("repeat", label)] = rep
            if li == 0:
                neg = []
                for s in ks:
                    if s["mode"] != "sample":
                        continue
                    al = eval_sequence(m, cap, s, args.k, args.batch, device, sp, args.draft_truncate, shift=1)
                    if al is not None:
                        neg.append(dict(summarize_seq(al, args.k), seq=s["seq"], domain=s["domain"], mode=s["mode"]))
                extra[("neg", label)] = neg
        del m
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    groups = sorted({(r["domain"], r["mode"]) for r in per[labels[0]]})
    for label in labels:
        res["results"][label] = {}
        for mode in sorted({g[1] for g in groups}):
            sel = [r for r in per[label] if r["mode"] == mode]
            res["results"][label][f"ALL|{mode}"] = aggregate(sel, args.k, args.bootstrap)
        for dom, mode in groups:
            sel = [r for r in per[label] if r["domain"] == dom and r["mode"] == mode]
            res["results"][label][f"{dom}|{mode}"] = aggregate(sel, args.k, args.bootstrap)
        res["per_seq"][label] = [{k: r[k] for k in ("seq", "domain", "mode", "toks", "steps", "n_anchor")}
                                 for r in per[label]]
    for label in labels[1:]:
        res["paired"][f"{label}/{labels[0]}"] = {}
        for key in res["results"][labels[0]]:
            dom, mode = key.split("|")
            sa = [r for r in per[labels[0]] if (dom == "ALL" or r["domain"] == dom) and r["mode"] == mode]
            sb = [r for r in per[label] if (dom == "ALL" or r["domain"] == dom) and r["mode"] == mode]
            res["paired"][f"{label}/{labels[0]}"][key] = paired(sa, sb, args.bootstrap)
    print_table(res, labels, args.k)

    code = 0
    if invalid:
        for msg in invalid:
            print("INVALID:", msg, flush=True)
        code = 2
    if args.check_known:
        code = max(code, check_known(res, per, extra, args, invalid)) if code != 2 else 2
    return res, code


def check_known(res, per, extra, args, invalid):
    want = args.known_labels.split(",")
    if len(want) != 2 or any(w not in res["results"] for w in want):
        print(f"INVALID: --check-known needs drafters labelled {want}", flush=True)
        return 2
    b, w = want
    dom = args.known_domain
    cells = {}
    for mode in ("sample", "greedy"):
        for lab, kind in ((b, "bf16"), (w, "w4a16")):
            c = res["results"][lab].get(f"{dom}|{mode}")
            if c is None or c["n_anchor"] < MIN_ANCHORS:
                print(f"INVALID: {lab} {dom}|{mode} has {0 if c is None else c['n_anchor']} anchors "
                      f"(need {MIN_ANCHORS})", flush=True)
                return 2
            cells[(mode, kind)] = c["tau_chain"]
    neg = extra.get(("neg", b)) or []
    neg_tau = aggregate(neg, args.k, 0)["tau_chain"] if neg else float("nan")
    rep = {}
    for lab, kind in ((b, "bf16"), (w, "w4a16")):
        rows = extra.get(("repeat", lab)) or []
        for mode in ("sample", "greedy"):
            sel = [r for r in rows if r["mode"] == mode]
            rep[(mode, kind)] = aggregate(sel, args.k, 0)["tau_chain"] if sel else float("nan")
    checks = []
    for key, (lo, hi) in KNOWN.items():
        v = cells[key]
        checks.append((f"{key[0]} {key[1]} tau_chain {v:.3f} in [{lo:.3f}, {hi:.3f}]", lo <= v <= hi))
    gap = cells[("sample", "w4a16")] / cells[("sample", "bf16")]
    checks.append((f"sampling gap w4a16/bf16 {gap:.4f} in [{GAP[0]:.3f}, {GAP[1]:.3f}]", GAP[0] <= gap <= GAP[1]))
    par = cells[("greedy", "w4a16")] / cells[("greedy", "bf16")]
    checks.append((f"greedy parity w4a16/bf16 {par:.4f} in [{PARITY[0]:.3f}, {PARITY[1]:.3f}]",
                   PARITY[0] <= par <= PARITY[1]))
    spread = max(abs(rep[k] - cells[k]) / cells[k] for k in cells)
    checks.append((f"repeat spread (half batch) {100 * spread:.3f}% < {100 * REPEAT_TOL:.1f}%", spread < REPEAT_TOL))
    pr = res["paired"].get(f"{w}/{b}", {}).get(f"{dom}|sample")
    res["check"] = dict(domain=dom, cells={f"{k[0]}|{k[1]}": v for k, v in cells.items()}, gap=gap, parity=par,
                        repeat_spread=spread, negative_control_tau=neg_tau, paired_sample=pr,
                        checks=[dict(name=n, ok=bool(o)) for n, o in checks])
    print(f"\n--check-known on {dom}:")
    for n, o in checks:
        print(f"  {'ok  ' if o else 'FAIL'} {n}")
    if pr:
        print(f"  info paired sample ratio {pr['ratio']:.4f} CI95 [{pr['ci95'][0]:.4f}, {pr['ci95'][1]:.4f}]")
    base = cells[("sample", "bf16")]
    if not (neg_tau <= (1 - NEG_DROP) * base):
        print(f"NEGATIVE CONTROL FAILED: target rows shifted by one give tau_chain {neg_tau:.3f} vs {base:.3f} "
              f"(must drop >= {100 * NEG_DROP:.0f}%)", flush=True)
        return 3
    print(f"  ok   negative control: shifted target rows tau_chain {neg_tau:.3f} vs {base:.3f}")
    ok = all(o for _, o in checks)
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


def print_table(res, labels, K):
    print()
    hdr = f"{'drafter':<10} {'domain|mode':<24} {'seqs':>4} {'anchors':>7} {'tau_chain':>9} {'CI95':>15} " \
          f"{'tau_anc':>7}  " + " ".join(f"acc@{j + 1:<2}" for j in range(K))
    print(hdr)
    for label in labels:
        for key, r in res["results"][label].items():
            ci = r["tau_chain_ci95"]
            cis = f"[{ci[0]:.3f},{ci[1]:.3f}]" if not math.isnan(ci[0]) else "-"
            print(f"{label:<10} {key:<24} {r['n_seq']:>4} {r['n_anchor']:>7} {r['tau_chain']:>9.3f} {cis:>15} "
                  f"{r['tau_anchor']:>7.3f}  " + " ".join(f"{a:6.3f}" for a in r["acc_at"]))
    for pk, d in res["paired"].items():
        print(f"\npaired tau_chain ratio {pk}:")
        for key, r in d.items():
            ci = r["ci95"]
            cis = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if not math.isnan(ci[0]) else "-"
            print(f"  {key:<24} {r['ratio']:.4f}  CI95 {cis}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------------------
# CPU self-test


def _oracle_capture(root, fx, m, k=1):
    """A K=1 capture whose target IS the drafter: row t's logits are the drafter's slot-1
    scores at anchor t (candidates only), the reference continues with their argmax. Any
    misalignment between anchors, slots, rows or prev tokens shows up as alpha < 1."""
    src = R.Capture(fx["capture"])
    d = os.path.join(root, "oracle")
    wr = R.CaptureWriter(d, dict(src.meta, sampling={"temperature": 1.0, "top_k": 0, "top_p": 1.0}))
    for s in src.seqs[:4]:
        x = src.seq(s)
        ids = x["ids"].clone()
        T, r0 = s["len"], s["prompt_len"]
        V = m.lm_head.shape[0]
        full = torch.full((T, V), float("-inf"))
        full[:, 0] = 0.0
        full[r0 - 1, 0] = float("-inf")
        full[r0 - 1, ids[r0]] = 0.0          # the prefill's own sample: reference token r0
        with torch.inference_mode():
            kv = m.context_kv(x["taps"], x["pos"])
            for t in range(r0, T - 1):
                a = torch.tensor([t])
                h = m.block_forward(kv, x["pos"], ids[a], x["pos"][a], k=k)[:, 1:]
                cand, un = m.candidates(h)
                sc = m.edge_scores(cand, un, h, ids[a][:, None]).float()[0, 0]
                row = torch.full((V,), float("-inf"))
                row[cand[0, 0]] = sc
                full[t] = row
                ids[t + 1] = cand[0, 0, sc.argmax()]
        ti, tv, lse = R.topk_rows(full, src.meta["topk"])
        wr.add({kk: s[kk] for kk in ("prompt_len", "start_pos", "domain", "think", "seed", "src")} | {"mode": s["mode"]},
               x["taps"], ids, ti, tv, lse)
    wr.close()
    return d


def selftest():
    torch.manual_seed(0)
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name} {detail}", flush=True)
        if not ok:
            fails.append(name)

    root = tempfile.mkdtemp(prefix="d2selftest_")
    print(f"selftest in {root}")
    # 1. packing against compressed-tensors
    for bits in (4, 8):
        q = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (5, 256), dtype=torch.int8)
        p = R.pack_int32(q, bits)
        check(f"pack/unpack roundtrip int{bits}", torch.equal(R.unpack_int32(p, bits, q.shape), q))
        try:
            from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
        except Exception:
            try:
                from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
            except Exception:
                pack_to_int32 = None
        if pack_to_int32 is None:
            print("  skip: compressed_tensors not importable")
        else:
            check(f"pack == compressed_tensors.pack_to_int32 int{bits}", torch.equal(pack_to_int32(q, bits, packed_dim=1), p))
    # 2. vLLM parity of the grouped conv and the selector's edge scores (CPU paths)
    fx = R.make_tiny_fixture(root)
    m = R.load_drafter(fx["bf16"], fx["target"])
    try:
        from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv, _score_edges
    except Exception as e:  # noqa: BLE001
        print(f"  skip: vLLM DFlash2 module not importable here ({type(e).__name__})")
        _grouped_conv = None
    if _grouped_conv is not None:
        conv = m.layers[0].attention_conv
        B, L, H = 3, 8, m.c.H
        x = torch.randn(B, L, H, dtype=torch.bfloat16)
        delta = torch.randn(B, L, conv.taps, conv.G, dtype=torch.bfloat16) * 0.1
        mine = conv._convolve(x, delta, 0, L).float()
        ref = _grouped_conv(x.reshape(B * L, H), delta.reshape(B * L, conv.taps, conv.G), conv.base_kernel[0],
                            L, conv.G, conv.group, conv.taps).float().reshape(B, L, H)
        err = ((mine - ref).norm() / ref.norm()).item()
        check("grouped conv vs vLLM _grouped_conv", err < 1e-2, f"rel {err:.2e}")
        h = torch.randn(B, L, H, dtype=torch.bfloat16)
        cand, un = m.candidates(h)
        anchor = torch.randint(0, m.c.vocab, (B,))
        mine = m.edge_scores_full(cand, un, h, anchor).float()
        sel = m.candidate_selector
        ref = _score_edges(sel.predecessor_codebook, sel.successor_codebook, cand, un,
                           sel.hidden_projection(h), anchor, m.c.top_k).float()
        err = ((mine - ref).norm() / ref.norm()).item()
        check("edge scores vs vLLM _score_edges", err < 1e-2, f"rel {err:.2e}")
    # 3. oracle capture: target == drafter at k=1 -> alpha == 1 in both modes
    od = _oracle_capture(root, fx, m)
    base = ["--capture", od, "--drafter", f"bf16={fx['bf16']}", "--target", fx["target"], "--k", "1",
            "--draft-truncate", "0", "--device", "cpu", "--bootstrap", "50"]
    res, code = run(parse(base))
    tau_s = res["results"]["bf16"]["ALL|sample"]["tau_anchor"]
    tau_g = res["results"]["bf16"]["ALL|greedy"]["tau_anchor"]
    check("oracle sample tau_anchor == 2", abs(tau_s - 2) < 2e-3 and code == 0, f"{tau_s:.5f}")
    check("oracle greedy tau_anchor == 2", tau_g > 1.95 and code == 0, f"{tau_g:.5f}")
    ch = res["results"]["bf16"]["ALL|sample"]
    ns = [s["len"] - 1 - s["prompt_len"] for s in R.Capture(od).seqs if s["mode"] == "sample"]
    want = sum(ns) / sum(-(-n // 2) for n in ns)               # every step accepts its one draft
    check("oracle chain tau (all accepted)", abs(ch["tau_chain"] - want) < 1e-2 * want,
          f"{ch['tau_chain']:.4f} vs {want:.4f}")
    cap = R.Capture(od)
    s0 = [s for s in cap.seqs if s["mode"] == "sample"][0]
    sp = dict(temperature=1.0, top_k=0, top_p=1.0)
    al = eval_sequence(m, cap, s0, 1, 8, "cpu", sp, False, shift=1)
    check("oracle negative control (shifted rows) drops alpha", float(al.mean()) < 0.9, f"{float(al.mean()):.3f}")
    # 4. renewal process: hand-computed case
    toks, steps = chain(np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]))
    check("chain bookkeeping (tokens == anchors)", abs(toks - 4) < 1e-9, f"tokens {toks} steps {steps}")
    # 5. full k=7 run on the random fixture, both drafters, and the batch-size repeat
    full = ["--capture", fx["capture"], "--drafter", f"bf16={fx['bf16']}", "--drafter", f"w4a16={fx['w4a16']}",
            "--k", "7", "--device", "cpu", "--bootstrap", "50"]
    r1, c1 = run(parse(full + ["--batch", "16"]))
    r2, c2 = run(parse(full + ["--batch", "5"]))
    check("k=7 run exit 0", c1 == 0 and c2 == 0, f"{c1} {c2}")
    d = max(abs(r1["results"][lb][k_]["tau_chain"] - r2["results"][lb][k_]["tau_chain"]) / r1["results"][lb][k_]["tau_chain"]
            for lb in r1["results"] for k_ in r1["results"][lb])
    check("batch-size repeat spread < 0.5%", d < REPEAT_TOL, f"{100 * d:.4f}%")
    check("paired ratio present", "w4a16/bf16" in r1["paired"])
    # 6. --check-known on a capture without the known domain -> INVALID (exit 2)
    _, c3 = run(parse(full + ["--check-known", "--known-domain", "dom_a"]))
    check("--check-known on a too-small capture is INVALID (2)", c3 == 2, f"exit {c3}")
    print(f"\nselftest {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    return 0 if not fails else 1


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture")
    ap.add_argument("--drafter", action="append", default=[], help="LABEL=DIR (repeatable; first is the baseline)")
    ap.add_argument("--target", help="target dir for embed/lm_head (default: the capture's)")
    ap.add_argument("--k", type=int, default=7, help="draft tokens per step (the drafter block - 1)")
    ap.add_argument("--batch", type=int, default=64, help="anchors per drafter forward")
    ap.add_argument("--domains", default="")
    ap.add_argument("--modes", default="")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-k", dest="top_k", type=int)
    ap.add_argument("--top-p", dest="top_p", type=float)
    ap.add_argument("--draft-truncate", type=int, default=1, help="VLLM_DFLASH2_DRAFT_TOPK_TOPP (default on)")
    ap.add_argument("--device")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out")
    ap.add_argument("--check-known", action="store_true")
    ap.add_argument("--known-domain", default="real_think")
    ap.add_argument("--known-labels", default="bf16,w4a16")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if not a.selftest and (not a.capture or not a.drafter):
        ap.error("--capture and at least one --drafter LABEL=DIR are required")
    if any("=" not in d for d in a.drafter):
        ap.error("--drafter takes LABEL=DIR")
    return a


def main():
    a = parse()
    if a.selftest:
        sys.exit(selftest())
    res, code = run(a)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    sys.exit(code)


if __name__ == "__main__":
    main()
