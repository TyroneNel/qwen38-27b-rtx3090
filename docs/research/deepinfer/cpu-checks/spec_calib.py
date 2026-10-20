import numpy as np, spec_varlen_sim as v  # sibling import: run with python3 -E, not -I
# find (m0, decay) giving pos0 ~0.76 and tok/step(k=7) ~3.2 (DFlash2, optimizations.md:395); MTP k=3 ~2.7 pos0 ~0.75
for m0 in (0.74, 0.77, 0.80, 0.83):
    for decay in (0.85, 0.88, 0.91, 0.94):
        a, acc, c = v.gen(20000, 1, 7, m0, decay, 0.8, 1.0, 4.0, 0.0)
        print(m0, decay, "pos0 %.3f" % acc[:, 0, 0].mean(), "k7 %.2f k3 %.2f" % (v.mean_tps(acc, 7), v.mean_tps(acc, 3)))
