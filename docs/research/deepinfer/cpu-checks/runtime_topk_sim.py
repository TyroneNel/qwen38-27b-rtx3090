# Illustrative CPU check: global top-k verify allocation vs best uniform k.
# Assumptions (NOT measured): per-request geometric survival a_i^j, a_i~Beta(6,4) (mean .6);
# step cost = 1 + r*(total_q-N) in units of the no-draft step; r = extra cost per verified token.
import numpy as np
rng=np.random.default_rng(0); K=15
def run(N,r,trials=400):
    gu=gg=0
    for _ in range(trials):
        a=rng.beta(6,4,N); surv=a[:,None]**np.arange(1,K+1)   # N x K
        best_u=max((N+surv[:,:k].sum())/(1+r*N*k) for k in range(0,K+1))
        s=np.sort(surv.ravel())[::-1]; c=np.cumsum(s)
        best_g=max([N/(1.0)]+[(N+c[m-1])/(1+r*m) for m in range(1,N*K+1)])
        gu+=best_u; gg+=best_g
    return gg/gu-1
for r in (0.002,0.01,0.03):
    print("r=%.3f"%r, " ".join("N=%d:%+.1f%%"%(N,100*run(N,r)) for N in (1,4,16,64)))
