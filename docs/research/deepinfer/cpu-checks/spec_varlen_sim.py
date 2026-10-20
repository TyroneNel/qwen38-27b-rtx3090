"""CPU simulation: fixed-k verify vs confidence-scheduled (DSpark / vLLM adaptive_verification) verify.

Synthetic acceptance (not measured): per step a latent z_t = rho*z_{t-1} + noise shifts the logit of
every position's conditional acceptance; per position a_i ~ Beta around sigmoid(logit(m_i)+z_t).
Realised accept events ~ Bernoulli(a_i). The estimator sees c_i = sigmoid(logit(a_i) + N(0, sigma)).
Budget policy = vLLM 0.31 get_num_tokens (adaptive_verification.py:282-350): argmax over total drafts of
(num_reqs + cumsum(sorted survival)) / cost(total tokens), survival = cumprod(confidence), using the
PREVIOUS step's confidences (stale) for the budget and the CURRENT ones to distribute it across requests
(reallocate_drafts). Cost models are linear-in-tokens estimates built from HyperQwen measurements
(see spec.md); every output is an estimate.
Run: python3 -I spec_varlen_sim.py
"""
import numpy as np

rng = np.random.default_rng(0)
STEPS = 40000


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(x):
    return 1 / (1 + np.exp(-x))


def gen(steps, nreq, K, m0, decay, rho, zsd, conc, sigma):
    """Returns true cond. accept probs a, accept events acc, estimator confidences c: [steps, nreq, K]."""
    z = np.zeros((steps, nreq))
    eps = rng.normal(0, zsd * np.sqrt(1 - rho**2), (steps, nreq))
    for t in range(steps):
        z[t] = (rho * z[t - 1] if t else 0) + eps[t]
    m = m0 * decay ** np.arange(K)
    mu = sig(logit(m)[None, None, :] + z[:, :, None])
    a = rng.beta(mu * conc, (1 - mu) * conc)
    acc = rng.random(a.shape) < a
    c = sig(logit(a) + rng.normal(0, sigma, a.shape)) if sigma > 0 else a
    return a, acc, c


def lead(acc):
    return np.cumprod(acc, axis=-1).sum(-1)  # accepted drafts if all K verified


def run_fixed(acc, k, cost):
    nreq = acc.shape[1]
    toks = (1 + np.minimum(lead(acc), k)).sum()
    t = cost(nreq * (1 + k), nreq, k) * acc.shape[0]
    return toks / t * 1000


def run_adaptive(acc, c, cost, stale=True, kmax=None):
    steps, nreq, K = acc.shape
    kmax = K if kmax is None else kmax
    S = np.cumprod(c[:, :, :kmax], axis=-1)  # survival
    L = lead(acc[:, :, :kmax])
    toks = 0.0
    time = 0.0
    ns = []
    for t in range(steps):
        Sb = S[t - 1] if (stale and t) else S[t]
        sc = np.sort(Sb.ravel())[::-1]
        exp_tok = nreq + np.concatenate(([0.0], np.cumsum(sc)))
        costs = np.array([cost(nreq + T, nreq, kmax) for T in range(len(sc) + 1)])
        T = int(np.argmax(exp_tok / costs))
        # distribute T over requests with CURRENT survival (global top-T, prefixes since S is monotone)
        flat = S[t].ravel()
        if T > 0:
            thr_idx = np.argsort(-flat, kind="stable")[:T]
            win = np.zeros_like(flat, dtype=bool)
            win[thr_idx] = True
            n_r = win.reshape(nreq, kmax).sum(1)
        else:
            n_r = np.zeros(nreq, dtype=int)
        toks += (1 + np.minimum(L[t], n_r)).sum()
        time += costs[T]
        ns.append(n_r.mean())
    return toks / time * 1000, float(np.mean(ns))


def lin_cost(F, c_tok, draft_ms_per_k=0.0):
    # step ms = F + c_tok * query tokens (+ sequential drafter cost per drafted position, MTP)
    return lambda ntok, nreq, kdraft: F + c_tok * ntok + draft_ms_per_k * kdraft


def mean_tps(acc, k):
    return float((1 + np.minimum(lead(acc), k)).mean())


def scenario(name, nreq, K, m0, decay, cost, ks, rho=0.8, zsd=1.0, conc=4.0, sigmas=(0.0, 1.0)):
    print(f"\n== {name} (nreq={nreq}, K={K}, m0={m0}, decay={decay}, rho={rho}, zsd={zsd}, conc={conc})")
    base = None
    for sigma in sigmas:
        a, acc, c = gen(STEPS if nreq == 1 else STEPS // 4, nreq, K, m0, decay, rho, zsd, conc, sigma)
        if base is None:
            fixed = {k: run_fixed(acc, k, cost) for k in ks}
            for k in ks:
                print(f"  fixed k={k}: tok/step/req {mean_tps(acc, k):.2f}  tok/s {fixed[k]:.1f}")
            base = fixed
        for stale in (True, False):
            tps, nbar = run_adaptive(acc, c, cost, stale=stale)
            best_k = max(base, key=base.get)
            print(
                f"  adaptive sigma={sigma} stale={stale}: tok/s {tps:.1f}  mean drafts verified {nbar:.2f}"
                f"  vs best fixed k={best_k} {100 * (tps / base[best_k] - 1):+.1f}%"
                f"  vs k={ks[-1]} {100 * (tps / base[ks[-1]] - 1):+.1f}%"
            )


if __name__ == "__main__":
    # Self-check: with a flat cost the policy must verify everything; with zero survival, nothing.
    _, acc, c = gen(200, 1, 7, 0.75, 1.0, 0.0, 0.0, 4.0, 0.0)
    flat = lambda ntok, nreq, k: 10.0
    _, nbar = run_adaptive(acc, c, flat, stale=False)
    assert abs(nbar - 7) < 1e-9, nbar
    _, nbar0 = run_adaptive(acc, np.zeros_like(c), lin_cost(10, 1), stale=False)
    assert nbar0 == 0
    print("self-check ok")

    # Acceptance calibrated with calib.py: DFlash2 m0=0.80 decay=0.86 -> pos0 ~0.77, k=7 ~3.2 tok/step
    # (optimizations.md:395: 3.14/3.34, ~75/78%); MTP m0=0.77 decay=0.95 -> k=3 ~2.65 (D measured ~2.7);
    # DFlash2 at long context m0=0.70 decay=0.85 -> k=7 ~2.6 (long-context.md:199-207: 2.0-3.1).
    DF = dict(m0=0.80, decay=0.86)
    # B: DFlash2 k=7, ~4k ctx, batch 1. step(8 tok)=25.9 ms (optimizations.md:358-360);
    # marginal 0.19 ms/token (+6% for 8->16 at 1.5k, lookup patch:959) -> F = 25.9 - 8*0.19.
    for c_tok in (0.19, 0.4):
        scenario(f"B DFlash2 4k c_tok={c_tok}", 1, 7, DF["m0"], DF["decay"], lin_cost(25.9 - 8 * c_tok, c_tok), (3, 5, 7))

    # D: MTP k=3 at ~150k, batch 1; step ~27.6 ms at 4 tokens (inferred from ~97 tok/s at ~2.7 tok/step);
    # per-position attention 0.6-1.8 ms (inferred from long-context.md:534-545 scaled, see spec.md).
    for c_tok in (0.6, 1.2, 1.8):
        scenario(f"D MTP 150k c_tok={c_tok}", 1, 3, 0.77, 0.95, lin_cost(27.6 - 4 * c_tok, c_tok), (1, 2, 3))
    # D': MTP drafting 5 (each extra MTP pass ~0.7 ms, optimizations.md:83) and verifying 1..5 adaptively.
    scenario("D' MTP 150k draft5 c_tok=1.2", 1, 5, 0.77, 0.95,
             lin_cost(27.6 - 4 * 1.2 - 3 * 0.7, 1.2, 0.7), (3, 4, 5))

    # E: DFlash2 k=7 at ~240k (KVarN), 67 tok/s measured; tok/step ~2.4 assumed -> step ~35.8 ms at 8 tokens.
    for c_tok in (1.0, 2.0):
        scenario(f"E DFlash2 240k c_tok={c_tok}", 1, 7, 0.70, 0.85,
                 lin_cost(1000 / 67 * 2.4 - 8 * c_tok, c_tok), (3, 5, 7))

    # Few users: DFlash2 at 4 streams, 4k: step 49.1 ms at 4x8 tokens (optimizations.md:358-360).
    # c_tok chosen so that fixed k=3 beats k=7 by ~8.5% (DYNAMIC_K measured +7.7..9.2%, start_qwen.sh:678).
    for c_tok in (0.7, 0.8, 0.9):
        scenario(f"C4 DFlash2 4k c_tok={c_tok}", 4, 7, DF["m0"], DF["decay"],
                 lin_cost(49.1 - 32 * c_tok, c_tok), (3, 7), sigmas=(1.0,))
