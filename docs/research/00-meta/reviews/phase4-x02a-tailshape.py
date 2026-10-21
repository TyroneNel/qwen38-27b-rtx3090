# Phase-4 review of X-02a: does "fitted vs DYNAMIC_K = 0.0% at N=4/8" survive a different split of the
# measured tail mass E7-E3 across positions 3..6? X-02a splits it by spec-5's within-group shape, which puts
# a cliff at position 3 (a3/a2 = 0.45). Here the tail is smooth: conditional acceptance q from position 3 on.
# Inputs are X-02a's measured cells (inputs.md section 4, Q1): E1=1.811, E3=2.862, E7 3.36..3.67 (cell spread),
# V(4,3)=42.3, V(4,7)=57.0, V(8,3)=61.8, V(8,1)=48.2. Estimated: V linear in k; V(8,k>3) slope unmeasured.
E1, E3 = 1.811, 2.862
a0 = E1 - 1
# a1 + a2 = E3 - E1 with a constant conditional p: a0*(p + p^2) = E3 - E1
p = next(x / 1000 for x in range(1, 1000) if a0 * (x / 1000 + (x / 1000) ** 2) >= E3 - E1)
a2 = a0 * p * p

def q_for(tail):  # a2*(q + q^2 + q^3 + q^4) = tail
    return next(x / 1000 for x in range(1, 2000) if a2 * sum((x / 1000) ** i for i in range(1, 5)) >= tail)

V4 = lambda k: 42.3 + (57.0 - 42.3) / 4 * (k - 3)
print(f"a0={a0:.3f} p={p:.3f} a2={a2:.3f}")
print(f"N=4: +5% for k=4 needs a3 >= {1.05 * E3 / V4(3) * V4(4) - E3:.3f}")
for E7 in (3.36, 3.508, 3.67):
    q = q_for(E7 - E3)
    a = [a2 * q ** i for i in range(1, 5)]          # a3..a6
    Ek = {3: E3, 4: E3 + a[0], 5: E3 + a[0] + a[1], 6: E3 + sum(a[:3]), 7: E7}
    g4 = max(Ek[k] / V4(k) for k in Ek) / (E3 / V4(3)) - 1
    k4 = max(Ek, key=lambda k: Ek[k] / V4(k))
    row = f"E7={E7:.3f} q={q:.3f} a3={a[0]:.3f} | N=4 best k={k4} gain vs k=3 {g4 * 100:+.1f}%"
    for s8 in (3.675, 5.0, 6.8, 7.35):              # N=8 ms per extra k beyond 3 (unmeasured)
        V8 = lambda k: 61.8 + s8 * (k - 3)
        g8 = max(Ek[k] / V8(k) for k in Ek) / (E3 / V8(3)) - 1
        row += f" | N=8 slope {s8}: {g8 * 100:+.1f}%"
    print(row)
    assert g4 < 0.05  # the H02' bar is not reached at N=4 under a smooth tail
