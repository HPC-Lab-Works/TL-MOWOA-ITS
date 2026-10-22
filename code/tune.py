"""Parameter tuning for TL-MOWOA on tuning seeds 101-103 only (disjoint from
the evaluation seeds 1-30)."""
import sys, json, itertools, numpy as np
from multiprocessing import Pool
import run as R, metrics as Mx

SRC = sys.argv[1]
grid = [{"p_tl": p, "tau_c": tc, "tau_d": td}
        for p in (0.1, 0.3, 0.5) for (tc, td) in ((1.0, 0.2), (2.0, 0.25), (1e9, 0.0))]
jobs = [("TL-MOWOA", s, SRC, g) for g in grid for s in (101, 102, 103)]
with Pool(2) as pool:
    res = pool.map(R.job, jobs)
fronts = [r["F"][r["V"] == 0] for r in res]
Ps, lo, hi = Mx.reference_front(fronts)
S = Mx.Scorer(Ps, lo, hi)
for gi, g in enumerate(grid):
    rr = res[gi * 3:(gi + 1) * 3]
    hv = [S.HV(r["F"][r["V"] == 0]) for r in rr]
    igd = [S.IGD(r["F"][r["V"] == 0]) for r in rr]
    print(json.dumps(g), "HV %.4f IGD %.4f exploit %s" % (np.mean(hv), np.nanmean(igd), [r["info"]["exploit_gens"] for r in rr]))
