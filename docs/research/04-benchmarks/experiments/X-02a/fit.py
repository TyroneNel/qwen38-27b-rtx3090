#!/usr/bin/env python3
"""X-02a / H02': fit a k-per-concurrency table by maximising E(k) / V(N,k), back-test it, compare to DYNAMIC_K.

stdlib only.  CPU only.  Run: python3 fit.py > fit.out
Every number in CELLS/AB/README_CURVE/SPEC5 is an input copied from raw/ (see inputs.md for path:line).
Everything computed here is ESTIMATED.

Model (ponytail: piecewise-linear interpolation of measured cells; no parametric fit,
 because the data are 4 k-values x 4 N-values and every extra parameter would be unidentified):
  rate(N,k) = E(k) / (V(N,k) + k*D)            tokens per ms per stream
  E(k)   = 1 + sum_{i<k} a_i                   a_i = per-position acceptance (two profiles, see below)
  V(N,k) = measured ms/pass cell at k in {0,1,3,7}; linear in k between cells (shape knob), linear in N between cells
  D      = extra drafter cost per position; 0 by default (ms/pass already includes drafting, and the
           DFlash drafter proposes its fixed block in one pass)
"""
import json

# ---------------------------------------------------------------- inputs (measured)
# (N,k) -> (ms/pass, tok/step, tok/pass, decode agg tok/s, running streams)
# Q1 = batch 3 Q1, int8 CTX=fast, MAX_SEQS=8 (raw/batch3-*), Q2 = batch 4 Q2, MAX_SEQS=16 (raw/batch4-*)
Q1 = {
    (1, 7): (29.5, 3.36, 3.4, 113.9, 1), (2, 7): (38.4, 3.46, 6.9, 179.6, 2), (4, 7): (57.0, 3.51, 13.6, 239.2, 4),
    (8, 7): (45.9, 3.45, 8.2, 178.5, 6),  # tail row (6 of 8 running): excluded
    (1, 3): (28.9, 2.91, 2.9, 100.4, 1), (2, 3): (32.0, 2.85, 5.6, 175.9, 2), (4, 3): (42.3, 2.89, 11.4, 268.4, 4),
    (8, 3): (61.8, 2.80, 21.7, 350.1, 8),
    (1, 1): (28.1, 1.83, 1.8, 65.3, 1), (2, 1): (30.0, 1.80, 3.6, 119.8, 2), (4, 1): (35.2, 1.80, 7.2, 203.9, 4),
    (8, 1): (48.2, 1.82, 14.4, 299.4, 8),
}
Q2 = {
    (1, 7): (32.7, 3.45, 3.4, 105.4, 1), (2, 7): (42.5, 3.60, 7.1, 167.4, 2), (4, 7): (60.3, 3.67, 14.4, 239.1, 4),
    (1, 3): (29.4, 2.98, 3.0, 100.9, 1), (2, 3): (33.2, 2.86, 5.7, 170.0, 2), (4, 3): (42.2, 2.82, 11.1, 263.6, 4),
    (8, 3): (63.9, 2.79, 21.5, 337.1, 8),
    (1, 1): (29.7, 1.81, 1.8, 60.9, 1), (2, 1): (30.9, 1.80, 3.6, 115.7, 2), (4, 1): (35.8, 1.82, 7.2, 202.3, 4),
    (8, 1): (49.1, 1.81, 14.5, 294.9, 8),
    (1, 0): (25.0, 1.0, 1.0, 40.1, 1), (2, 0): (25.8, 1.0, 2.0, 77.5, 2), (4, 0): (27.9, 1.0, 4.0, 143.2, 4),
    (8, 0): (31.9, 1.0, 8.0, 250.7, 8), (16, 0): (43.3, 1.0, 16.0, 369.2, 16),
}
# DYNAMIC_K A/B, 4 interleaved boots, bf16 FLASH_ATTN, MAX_SEQS=8 (raw/dynamic-k-ab-salted-report.md:31-47)
AB = {"k7_N1_ms": 28.95, "k7_N1_E": 3.45, "k7_N4_ms": 56.63, "k7_N4_E": 3.43, "k7_N4_agg": 241.4,
      "dyn_N4_ms": 43.15, "dyn_N4_E": 2.81, "dyn_N4_agg": 260.0, "gain_reported": (7.7, 9.2)}
README_CURVE = {1: 25.9, 2: 32.8, 4: 49.1}            # single-user/README.md:116, k=7, ms/pass
README_AGG = {1: 137.0, 2: 225.0, 4: 309.0}           # single-user/README.md:115, k=7
# spec-5 per-position realised accept, T=0.7 thinking off (spec-5.md:40-44); positions 4,5 not published
SPEC5_KNOWN = {0: 0.687, 1: 0.511, 2: 0.404, 3: 0.325, 6: 0.190}
SPEC5_E7 = 3.611                                       # spec-5.md:33
DYNAMIC_K = [(1, 3, 7), (4, 512, 3)]                   # single-user/start_qwen.sh:678, README.md:434
KS = range(0, 8)


def mean(x):
    x = list(x)
    return sum(x) / len(x)


# ---------------------------------------------------------------- acceptance profiles
def spec5_profile():
    """a_0..a_6.  a4,a5 unpublished: solved from E7 (sum constraint), split by the geometric trend of
    the conditional accept (0.744,0.790,0.805,...,0.841)."""
    a = dict(SPEC5_KNOWN)
    rest = SPEC5_E7 - 1 - sum(a.values())             # a4 + a5
    r4 = 0.82                                          # conditional accept at pos 4 (between 0.805 and 0.841)
    a4 = a[3] * r4
    a5 = rest - a4
    a[4], a[5] = a4, a5
    return [a[i] for i in range(7)]


def pooled_E():
    """Measured tok/step pooled over Q1+Q2 boots (NOT the DYNAMIC_K A/B boots): k=1,3 over N<=8, k=7 over N<=4."""
    cells = list(Q1.items()) + list(Q2.items())
    E = {0: 1.0}
    for k, nmax in ((1, 8), (3, 8), (7, 4)):
        E[k] = mean(v[1] for (n, kk), v in cells if kk == k and n <= nmax)
    return E


def ladder_profile(E=None, tail=1.0):
    """a_i matched to measured E1,E3,E7: increments split by the spec-5 within-group shape (estimated)."""
    E = E or pooled_E()
    s5 = spec5_profile()
    a = [E[1] - 1]
    inc = E[3] - E[1]
    a += [inc * s5[1] / (s5[1] + s5[2]), inc * s5[2] / (s5[1] + s5[2])]
    inc = (E[7] - E[3]) * tail
    g = sum(s5[3:7])
    a += [inc * s5[i] / g for i in range(3, 7)]
    return a


def E_of(a, k):
    return 1 + sum(a[:k])


# ---------------------------------------------------------------- cost model V(N,k)
def anchors(base="Q1", v0mult=1.0):
    """anchor[N][k] ms for k in {0,1,3,7}, N in {1,2,4,8,16}.  k>=1 from Q1 (8-seat boot) by default;
    k=0 from Q2-off (16-seat boot, biased high; v0mult is the sensitivity knob).  Tail rows excluded;
    k=7 at N>=8 and k>=1 at N=16 extrapolated (estimated)."""
    src = Q1 if base == "Q1" else Q2 if base == "Q2" else {k: tuple(mean(x) for x in zip(Q1[k], Q2[k])) for k in Q1 if k in Q2}
    A = {n: {} for n in (1, 2, 4, 8, 16)}
    for (n, k), v in src.items():
        if v[4] == n and n <= 8 and k in (1, 3, 7):
            A[n][k] = v[0]
    for (n, k), v in Q2.items():
        if k == 0:
            A[n][0] = v[0] * v0mult
    # k=7 at N=8: V(8,3) + 4*g, per-position slope g scaled with N from N=4
    g4 = (A[4][7] - A[4][3]) / 4
    A[8][7] = A[8][3] + 4 * g4 * 8 / 4
    # N=16: extend k=1,3,7 linearly in N from the N=4..8 slope
    for k in (1, 3, 7):
        A[16][k] = A[8][k] + (A[8][k] - A[4][k]) / 4 * 8
    return A


def interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    i = len(xs) - 1
    return ys[i] + (x - xs[i]) * (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1])


def V(N, k, A, shape="linear"):
    """ms/pass at N residents, verify depth k."""
    Ns = [1, 2, 4, 8, 16]

    def at(kk):
        return interp(Ns, [A[n][kk] for n in Ns], N)
    if k in (0, 1, 3, 7):
        return at(k)
    lo, hi = (1, 3) if k < 3 else (3, 7)
    t = (k - lo) / (hi - lo)
    if shape == "convex":      # cost concentrated at high k (stair near 16+ rows)
        t = t * t
    elif shape == "concave":
        t = t ** 0.5
    return at(lo) + t * (at(hi) - at(lo))


def spec_c():
    """decode-agg / (N * tok/step / ms): ragged-batch factor, measured on k>=3 steady cells."""
    r = [v[3] / (n * v[1] / v[0] * 1000) for src in (Q1, Q2) for (n, k), v in src.items() if k >= 3 and v[4] == n]
    return mean(r)


C_SPEC = spec_c()


def agg(N, k, a, A, D=0.0, shape="linear"):
    """predicted decode aggregate tok/s (estimated)."""
    c = 1.0 if k == 0 else C_SPEC
    return 1000 * N * E_of(a, k) / (V(N, k, A, shape) + k * D) * c


def best_k(N, a, A, kmax=7, D=0.0, shape="linear", allow_off=True):
    ks = [k for k in KS if k <= kmax and (allow_off or k > 0)]
    return max(ks, key=lambda k: agg(N, k, a, A, D, shape))


def dyn_k(N, table=DYNAMIC_K):
    for lo, hi, k in table:
        if lo <= N <= hi:
            return k
    raise ValueError(N)


def table_ranges(kN):
    """{N:k} -> DYNAMIC_K ranges, last range extended to 512."""
    Ns = sorted(kN)
    out, start = [], Ns[0]
    for i, n in enumerate(Ns):
        if i + 1 == len(Ns) or kN[Ns[i + 1]] != kN[n]:
            out.append([start, n, kN[n]])
            if i + 1 < len(Ns):
                start = Ns[i + 1]
    out[-1][1] = 512
    return out


# ---------------------------------------------------------------- back-test
def backtest(a, A, label, shape="linear"):
    """Predict the DYNAMIC_K gain at N=4 (k=3 vs k=7) with V(4,3) HELD OUT: interpolated from the (4,1),(4,7) cells."""
    v1, v7 = A[4][1], A[4][7]
    t = 2 / 6
    t = t * t if shape == "convex" else t ** 0.5 if shape == "concave" else t
    v3_pred = v1 + (v7 - v1) * t
    ratio_v = v3_pred / v7
    gain = (E_of(a, 3) / E_of(a, 7)) / ratio_v - 1
    lo, hi = AB["gain_reported"]
    ok = lo - 3 <= gain * 100 <= hi + 3
    return dict(label=label, V3_pred=v3_pred, V3_meas_Q1=Q1[(4, 3)][0], V3_meas_AB=AB["dyn_N4_ms"], Vratio=ratio_v,
                E3=E_of(a, 3), E7=E_of(a, 7), gain_pct=gain * 100, passed=ok)


def main():
    out = []
    P = out.append
    s5 = spec5_profile()
    E = pooled_E()
    lad = ladder_profile(E)
    A = anchors("Q1")
    P("== inputs digest ==")
    P(f"pooled measured E (Q1+Q2 boots): k1={E[1]:.3f} k3={E[3]:.3f} k7={E[7]:.3f}   (DYNAMIC_K A/B, N=4: k7={AB['k7_N4_E']}, k3={AB['dyn_N4_E']}, ratio {AB['dyn_N4_E']/AB['k7_N4_E']:.3f}; pooled ratio {E[3]/E[7]:.3f})")
    P(f"spec-5 profile a_0..a_6 = {[round(x, 3) for x in s5]}  E3={E_of(s5, 3):.3f} E7={E_of(s5, 7):.3f}  E3/E7={E_of(s5, 3)/E_of(s5, 7):.3f}")
    P(f"ladder-matched profile  = {[round(x, 3) for x in lad]}  E1={E_of(lad, 1):.3f} E3={E_of(lad, 3):.3f} E7={E_of(lad, 7):.3f}")
    P(f"ragged-batch factor c (agg / N*tok_step/ms) = {C_SPEC:.3f}")
    P("anchor ms/pass (Q1 for k>=1, Q2-off for k=0; '*' = extrapolated):")
    for n in (1, 2, 4, 8, 16):
        P(f"  N={n:2d} " + " ".join(f"k{k}={A[n][k]:6.1f}{'*' if (n == 16 and k) or (n == 8 and k == 7) else ' '}" for k in (0, 1, 3, 7)))

    P("\n== (a) BACK-TEST: DYNAMIC_K gain at N=4, k=3 vs k=7, V(4,3) held out ==")
    P(f"reported gain {AB['gain_reported'][0]}-{AB['gain_reported'][1]}% (single-user/README.md:434); raw salted A/B +7.7% (agg {AB['k7_N4_agg']} -> {AB['dyn_N4_agg']}); pass window = reported range +-3 pts = [4.7, 12.2]")
    bt = {}
    for label, a in (("spec-5 per-position a_i (as specified)", s5), ("ladder-matched a_i (pooled measured E1/E3/E7)", lad)):
        r = backtest(a, A, label)
        bt[label] = r
        P(f"  {label:48s} V3_pred={r['V3_pred']:.1f} (meas Q1 {r['V3_meas_Q1']}, A/B {r['V3_meas_AB']})  Vratio={r['Vratio']:.3f}  E3/E7={r['E3']/r['E7']:.3f}  gain={r['gain_pct']:+.1f}%  {'PASS' if r['passed'] else 'FAIL'}")
    Aq2 = anchors("Q2")
    r = backtest(lad, Aq2, "ladder-matched, V from Q2 (16-seat boot)")
    P(f"  {r['label']:48s} V3_pred={r['V3_pred']:.1f} (meas Q2 {Q2[(4,3)][0]})  Vratio={r['Vratio']:.3f}  gain={r['gain_pct']:+.1f}%  {'PASS' if r['passed'] else 'FAIL'}")
    obs = (AB['dyn_N4_E'] / AB['k7_N4_E']) / (AB['dyn_N4_ms'] / AB['k7_N4_ms']) - 1
    P(f"  sanity: A/B step+acceptance cells give {obs*100:+.1f}% (decode-agg +7.7%); Q1 boot-time k=3 vs k=7 agg {Q1[(4,3)][3]/Q1[(4,7)][3]*100-100:+.1f}%, Q2 {Q2[(4,3)][3]/Q2[(4,7)][3]*100-100:+.1f}%")
    s5_oracle = (E_of(s5, 3) / E_of(s5, 7)) / (AB['dyn_N4_ms'] / AB['k7_N4_ms']) - 1
    P(f"  spec-5 a_i with the A/B's OWN measured V ratio ({AB['dyn_N4_ms']/AB['k7_N4_ms']:.3f}): {s5_oracle*100:+.1f}%  -> the miss is acceptance shape, not the cost model")
    # README step curve consistency: scale of V7 only (cancels in the gain)
    P("  README step curve (25.9/32.8/49.1) vs Q1 k=7 cells: " + ", ".join(f"N={n}: {README_CURVE[n]/Q1[(n,7)][0]:.2f}x" for n in (1, 2, 4)) + "  (older/faster image; scale cancels in the gain ratio)")
    P("  README implied tok/step = agg*ms/N/1000: " + ", ".join(f"N={n}: {README_AGG[n]*README_CURVE[n]/n/1000:.2f}" for n in (1, 2, 4)))

    # ------------------------------------------------------------ fit
    Ns = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16]
    P("\n== (b) FITTED TABLE vs DYNAMIC_K  (DFLASH_TOKENS=7 boot, k in 0..7, native-OFF allowed; rows N>8 extrapolated) ==")
    fitted = {}
    for name, a in (("ladder-matched", lad), ("spec-5", s5)):
        P(f"-- acceptance profile: {name}")
        P("  N | fit k | DYNAMIC_K k | agg(fit) | agg(dyn) | gain  | agg(k=7) | agg(off) | note")
        fk = {}
        for n in Ns:
            k = best_k(n, a, A)
            kd = dyn_k(n)
            gf, gd = agg(n, k, a, A), agg(n, kd, a, A)
            fk[n] = k
            note = ("extrapolated" if n > 8 else "") + ("; seat-limited on a k=7 boot (state pages)" if n >= 5 else "")
            P(f"  {n:2d} | {k:5d} | {kd:11d} | {gf:8.1f} | {gd:8.1f} | {100*(gf/gd-1):+5.1f}% | {agg(n,7,a,A):8.1f} | {agg(n,0,a,A):8.1f} | {note}")
        fitted[name] = fk
    # tables over the dense N range 1..16
    tabs = {}
    for name, a in (("ladder-matched", lad), ("spec-5", s5)):
        kN = {n: best_k(n, a, A) for n in range(1, 17)}
        tabs[name] = table_ranges(kN)
        P(f"fitted DYNAMIC_K ({name}, N=1..16): {json.dumps(tabs[name])}")
    P(f"existing DYNAMIC_K (single-user/start_qwen.sh:678): {json.dumps([list(t) for t in DYNAMIC_K])}")

    P("\n== (c) PREDICTED AGGREGATES, N=1/2/4/8 (estimated; ladder-matched a_i; tok/s) ==")
    P("DFLASH_TOKENS=7 boot, bf16-like base = Q1 cells.  N=8: seat-limited (5-7 resident at 4k prompts) -> compute-only figure, measured tail rows 176.8-193.6")
    tab = tabs["ladder-matched"]
    rows = {}
    for n in (1, 2, 4, 8):
        kf = best_k(n, lad, A)
        rows[n] = (agg(n, 7, lad, A), agg(n, dyn_k(n), lad, A), agg(n, kf, lad, A), kf)
        P(f"  N={n}: fixed k=7 {rows[n][0]:6.1f} | DYNAMIC_K(k={dyn_k(n)}) {rows[n][1]:6.1f} | fitted(k={kf}) {rows[n][2]:6.1f} | fit/dyn {100*(rows[n][2]/rows[n][1]-1):+.1f}%  | fit/k7 {100*(rows[n][2]/rows[n][0]-1):+.1f}%")
    T_FIT = [(1, 1, 7), (2, 2, 5), (3, 3, 4), (4, 512, 3)]
    T_BND = [(1, 2, 7), (3, 512, 3)]
    assert table_ranges({n: best_k(n, lad, A) for n in range(1, 17)}) == [list(t) for t in T_FIT]
    P("tables for X-02b' (ladder-matched, N=1..4,8; agg tok/s, estimated):")
    for nm, t in (("DYNAMIC_K  [[1,3,7],[4,512,3]]", DYNAMIC_K), ("fitted     [[1,1,7],[2,2,5],[3,3,4],[4,512,3]]", T_FIT), ("boundary   [[1,2,7],[3,512,3]]", T_BND)):
        P(f"  {nm:42s} " + "  ".join(f"N={n}: {agg(n, dyn_k(n, t), lad, A):6.1f}" for n in (1, 2, 3, 4, 8)))
    P("measured cells for comparison (decode agg): N=4 k7 239.2/239.1/241.4, k3 268.4/263.6/260.0(DYNAMIC_K); N=8 k3 boot 350.1/337.1, k=1 299.4/294.9, off 250.7")
    P("DFLASH_TOKENS=3 boot (seats ok at N=8): tables limited to k<=3")
    for n in (4, 8):
        kf = best_k(n, lad, A, kmax=3)
        P(f"  N={n}: k=3 {agg(n,3,lad,A):6.1f} | fitted(k<=3)={kf} {agg(n,kf,lad,A):6.1f} | off {agg(n,0,lad,A):6.1f}")

    # ------------------------------------------------------------ sensitivity
    P("\n== sensitivity: fitted/DYNAMIC_K gain (%) = argmax-table agg / DYNAMIC_K agg - 1 ==")
    P("grid: acceptance profile x V base (Q1/Q2) x k-shape (linear/convex/concave between measured k) x V0 (k=0 cost) x1 0.8/1.0/1.3.")
    P("D (extra drafter ms per position) is NOT in the grid: measured ms/pass already contains the drafter, adding k*D double counts; shown separately.")
    cases = []
    profiles = {"ladder": lad, "ladder tail x0.77": ladder_profile(E, 0.77), "ladder tail x0.9": ladder_profile(E, 0.9),
                "ladder tail x1.1": ladder_profile(E, 1.1), "ladder tail x1.25": ladder_profile(E, 1.25), "spec5": s5}
    # tail x0.77..x1.25 spans the cell-to-cell spread of measured E7 (3.36-3.67) around the pooled 3.51
    prof_ok = {pn: backtest(a, A, pn)["passed"] for pn, a in profiles.items()}
    for pn, a in profiles.items():
        for base in ("Q1", "Q2"):
            for shape in ("linear", "convex", "concave"):
                bt_ok = prof_ok[pn]
                for v0 in (0.8, 1.0, 1.3):
                    An = anchors(base, v0)
                    for n in (2, 3, 4, 8):
                        kf = best_k(n, a, An, shape=shape)
                        kd = dyn_k(n)
                        g = agg(n, kf, a, An, 0, shape) / agg(n, kd, a, An, 0, shape) - 1
                        cases.append((n, g * 100, pn, base, shape, v0, kf, bt_ok))
    for label, sel in (("ALL cases", lambda c: True), ("only acceptance profiles that PASS the back-test (linear V, Q1)", lambda c: c[7])):
        cs = [c for c in cases if sel(c)]
        P(f"-- {label}: {len(cs)//4} of {len(cases)//4} grid points")
        for n in (2, 3, 4, 8):
            cn = [c for c in cs if c[0] == n]
            top = max(cn, key=lambda c: c[1])
            med = sorted(c[1] for c in cn)[len(cn) // 2]
            ks = sorted(set(c[6] for c in cn))
            P(f"  N={n}: median {med:+.1f}%  max {top[1]:+.1f}% (profile={top[2]}, base={top[3]}, shape={top[4]}, V0x{top[5]}, fit k={top[6]});  fitted k seen: {ks}")
    for pn, a in profiles.items():
        P(f"  profile {pn:18s}: E7={E_of(a,7):.2f} back-test gain {backtest(a, A, pn)['gain_pct']:+.1f}% -> {'PASS' if prof_ok[pn] else 'FAIL'}")
    P("drafter cost D per position (ladder profile, Q1, linear):")
    for D in (0.0, 0.5, 1.0):
        P(f"  D={D}: N=4 fit k={best_k(4,lad,A,D=D)} N=8 fit k={best_k(8,lad,A,D=D)}; gain vs k=3 table N=4 {100*(agg(4,best_k(4,lad,A,D=D),lad,A,D)/agg(4,3,lad,A,D)-1):+.1f}%")
    off = [c for c in cases if c[6] == 0]
    P(f"cases where native-OFF (k=0) is the argmax at N in 2,3,4,8: {len(off)} of {len(cases)}")
    P("native-OFF break-even: smallest N in 1..16 where k=0 is the argmax (extrapolated above N=8), by V0 multiplier and kmax:")
    for v0 in (0.8, 1.0, 1.3, 1.5):
        An = anchors("Q1", v0)
        for kmax in (3, 7):
            ns = [n for n in range(1, 17) if best_k(n, lad, An, kmax=kmax) == 0]
            P(f"  V0x{v0} kmax={kmax}: " + (f"N>={ns[0]}" if ns else "never (N<=16)"))
    P("required E7-E3 (tail mass a_3..a_6) for fitted k>=5 at N=4 (ladder shape, linear V):")
    for t in (0.7, 1.0, 1.5, 2.0, 2.5):
        a = ladder_profile(E, t)
        P(f"  tail x{t}: E7={E_of(a,7):.2f} fit k at N=4: {best_k(4,a,A)}  gain vs DYNAMIC_K {100*(agg(4,best_k(4,a,A),a,A)/agg(4,3,a,A)-1):+.1f}%")

    # ------------------------------------------------------------ verdict numbers
    mx = {n: max(c[1] for c in cases if c[0] == n and c[7]) for n in (2, 3, 4, 8)}
    gain4 = rows[4][2] / rows[4][1] - 1
    gain8 = rows[8][2] / rows[8][1] - 1
    P("\n== verdict inputs ==")
    P(f"base-case fitted/DYNAMIC_K: N=4 {gain4*100:+.1f}%, N=8 {gain8*100:+.1f}%; back-test-consistent sensitivity max N=4 {mx[4]:+.1f}%, N=8 {mx[8]:+.1f}%, N=3 {mx[3]:+.1f}%")
    P(f"H02' bar (>=5% at N=4 or N=8): {'reached only at a corner of the back-test-consistent range (see max above; median 0)' if max(mx[4], mx[8]) >= 5 else 'NOT reached anywhere in the back-test-consistent sensitivity range'}")

    # ------------------------------------------------------------ self-check
    # 1. profiles reproduce their E anchors
    assert abs(E_of(lad, 1) - E[1]) < 1e-9 and abs(E_of(lad, 3) - E[3]) < 1e-9 and abs(E_of(lad, 7) - E[7]) < 1e-9
    assert abs(E_of(s5, 7) - SPEC5_E7) < 1e-9 and all(0 < x < 1 for x in s5) and s5[4] > s5[5] > 0
    # 2. V reproduces every anchor and is monotone in N and k (k>=1)
    for n in (1, 2, 4, 8):
        for k in (0, 1, 3, 7):
            assert abs(V(n, k, A) - A[n][k]) < 1e-9
        assert V(n, 1, A) <= V(n, 3, A) <= V(n, 7, A) + 1e-9
    assert V(3, 3, A) > V(2, 3, A) and V(3, 3, A) < V(4, 3, A)
    # 3. optimiser: k=7 at N=1 (ladder, spec-5); k=3 at N=4 (ladder) and N=8 (ladder, either boot)
    assert best_k(1, lad, A) == 7 and best_k(1, s5, A) == 7
    assert best_k(4, lad, A) == 3 and best_k(8, lad, A) <= 4
    # 4. table_ranges round-trips
    assert table_ranges({1: 7, 2: 7, 3: 7, 4: 3, 5: 3}) == [[1, 3, 7], [4, 512, 3]]
    # 5. model reproduces measured Q1/Q2 aggregates at the cells it is anchored on (within the c factor spread)
    for src in (Q1, Q2):
        for (n, k), v in src.items():
            if v[4] == n and n <= 8 and (k, src is Q1) != (0, True) and (n, k) != (8, 7):
                a = {1: [E[1] - 1], 3: lad, 7: lad, 0: lad}.get(k, lad)
                if k in (1, 3, 7, 0):
                    assert abs(agg(n, k, lad, A) / v[3] - 1) < 0.12, (n, k, agg(n, k, lad, A), v[3])
    # 6. back-test numbers
    assert not bt["spec-5 per-position a_i (as specified)"]["passed"]
    assert bt["ladder-matched a_i (pooled measured E1/E3/E7)"]["passed"]
    P("\nSELF-CHECK OK")
    print("\n".join(out))


if __name__ == "__main__":
    main()
