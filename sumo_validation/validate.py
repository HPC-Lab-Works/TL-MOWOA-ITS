"""SUMO validation of each run's best-compromise solution (closest to the ideal
point in the normalized objective space of the reference front)."""
import sys, glob, pickle, json, os, numpy as np
from multiprocessing import Pool
sys.path.insert(0, "/home/claude/tlmowoa")
import sumo_eval as S
import problem as P

RUNS, CURVES, OUT, SEEDS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
a, b = map(int, SEEDS.split("-"))
ALGS = sys.argv[5].split(",") if len(sys.argv) > 5 else ["TL-MOWOA", "MOWOA", "NSGA-III", "TL", "Hybrid", "GNN"]
cv = pickle.load(open(CURVES, "rb"))
lo, hi = cv["lo"], cv["hi"]


def compromise(r):
    m = r["V"] == 0
    if not m.any():
        return None, None
    F, X = r["F"][m], r["X"][m]
    j = np.argmin(np.linalg.norm((F - lo) / (hi - lo), axis=1))
    return X[j], F[j]


def work(job):
    alg, seed, x, fm = job
    res = S.simulate(x, seed=seed)
    res.update({"alg": alg, "seed": seed, "model_F": list(map(float, fm))})
    return res


jobs = []
for alg in ALGS:
    for s in range(a, b + 1):
        f = f"{RUNS}/{alg}_{s}.pkl"
        if not os.path.exists(f):
            continue
        x, fm = compromise(pickle.load(open(f, "rb")))
        if x is not None:
            jobs.append((alg, s, x, fm))
base = np.tile([100.0, 0.48, 0.0], P.NI)   # uniform fixed-time reference plan
jobs.append(("Uniform fixed-time plan", 0, base, P.evaluate(base, P.TARGET)[0][0]))
out = []
with Pool(2) as pool:
    for r in pool.imap_unordered(work, jobs):
        out.append(r)
        print(r["alg"], r["seed"], round(r["travel_time_min"], 3), round(r["fuel_L100km"], 3),
              round(r["co2_g_km"], 1), r["n_trips"], flush=True)
json.dump(out, open(OUT, "w"), indent=1, default=float)
