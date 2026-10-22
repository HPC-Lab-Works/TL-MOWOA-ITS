"""Round-2 development on tuning seeds 101-108 only. Every configuration
tried is logged to tuning_round2.json."""
import sys, json, numpy as np
from multiprocessing import Pool
import run as R, metrics as Mx

SRC, OUT = sys.argv[1], sys.argv[2]
base = {"p_tl": 0.1, "tau_c": 1.0, "tau_d": 0.2}
configs = {
    "C0 original": {},
    "C1 a_max=1": {"a_max": 1.0},
    "C2 TL-guided search": {"tl_search": True},
    "C3 adaptive p_TL": {"adaptive_ptl": True},
    "C4 adaptive + TL search": {"adaptive_ptl": True, "tl_search": True},
    "C5 adaptive + a_max=1": {"adaptive_ptl": True, "a_max": 1.0},
    "C6 adaptive + TL search + decaying refinement": {"adaptive_ptl": True, "tl_search": True, "loc_decay": True},
    "C7 adaptive (p0=0.3) + TL search": {"adaptive_ptl": True, "tl_search": True, "p_tl": 0.3},
}
seeds = range(101, 109)
jobs = [("TL-MOWOA", s, SRC, {**base, **c}) for c in configs.values() for s in seeds]
jobs += [("TL", s, SRC, base) for s in seeds]
with Pool(2) as pool:
    res = pool.map(R.job, jobs)
fr = [r["F"][r["V"] == 0] for r in res]
Ps, lo, hi = Mx.reference_front(fr)
S = Mx.Scorer(Ps, lo, hi)
names = list(configs) + ["TL-only (reference)"]
log = {}
for i, nm in enumerate(names):
    rr = res[i * len(seeds):(i + 1) * len(seeds)]
    hv = [S.HV(r["F"][r["V"] == 0]) for r in rr]; igd = [S.IGD(r["F"][r["V"] == 0]) for r in rr]
    log[nm] = {"cfg": {**base, **configs.get(nm, {})}, "hv_mean": float(np.mean(hv)), "hv_std": float(np.std(hv, ddof=1)),
               "igd_mean": float(np.nanmean(igd)), "igd_std": float(np.nanstd(igd, ddof=1))}
    print(f"{nm:48s} HV {np.mean(hv):.4f}±{np.std(hv, ddof=1):.4f}  IGD {np.nanmean(igd):.4f}±{np.nanstd(igd, ddof=1):.4f}", flush=True)
json.dump(log, open(OUT, "w"), indent=1)
