"""Run one (algorithm, seed) job on the target task; or a whole batch in parallel."""
import sys, os, pickle, time, json
import numpy as np
import problem as P
import algorithms as A
import tl

SRC = None
SRC_MODEL = None
CFG = {"p_tl": 0.3, "tau_c": 0.5, "tau_d": 0.12}


def _load(src_path):
    global SRC, SRC_MODEL
    if SRC is None:
        SRC = pickle.load(open(src_path, "rb"))
        cache = src_path.replace(".pkl", "_model.pt")
        import torch
        if os.path.exists(cache):
            m, sc, Xin = torch.load(cache, weights_only=False)
        else:
            m, sc, Xin = tl.pretrain_source(SRC)
            torch.save((m, sc, Xin), cache)
        SRC_MODEL = (m, sc, Xin)


def tm_for(seed):
    m, sc, Xin = SRC_MODEL
    return tl.TransferModule(SRC, m, sc, Xin, P.TARGET, seed=seed)


def job(args):
    alg, seed, src_path, cfg = args
    _load(src_path)
    t0 = time.time()
    if alg == "MOWOA":
        arch, rec, info = A.run_woa_family(P.TARGET, seed)
    elif alg == "Hybrid":
        arch, rec, info = A.run_woa_family(P.TARGET, seed, tm=tm_for(seed), tl_init=True, p_tl=cfg["p_tl"])
    elif alg == "TL-MOWOA":
        arch, rec, info = A.run_woa_family(P.TARGET, seed, tm=tm_for(seed), tl_init=True, tl_leader=True,
                                           controller=True, **cfg)
    elif alg == "TL-MOWOA-v2":     # revised: a decreases from 1 (no search-for-prey phase)
        arch, rec, info = A.run_woa_family(P.TARGET, seed, tm=tm_for(seed), tl_init=True, tl_leader=True,
                                           controller=True, a_max=1.0, **cfg)
    elif alg == "MOWOA-a1":        # ablation: same coefficient schedule, no transfer learning
        arch, rec, info = A.run_woa_family(P.TARGET, seed, a_max=1.0)
    elif alg == "Hybrid-a1":       # ablation: TL initialization only, same coefficient schedule
        arch, rec, info = A.run_woa_family(P.TARGET, seed, tm=tm_for(seed), tl_init=True, p_tl=cfg["p_tl"],
                                           a_max=1.0)
    elif alg == "TL":
        arch, rec, info = A.run_tl_only(P.TARGET, seed, tm_for(seed))
    elif alg == "NSGA-III":
        arch, rec, info = A.run_nsga3(P.TARGET, seed)
    elif alg == "GNN":
        arch, rec, info = A.run_gcn_saea(P.TARGET, seed)
    info = {k: v for k, v in info.items() if k not in ("ev", "p_tl_trace")}
    return {"alg": alg, "seed": seed, "X": arch.X, "F": arch.F, "V": arch.V,
            "evals": rec.evals, "fronts": rec.fronts, "best": np.array(rec.best),
            "info": info, "time": time.time() - t0}


if __name__ == "__main__":
    from multiprocessing import Pool
    src_path, out_dir, algs, seeds = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
    a, b = map(int, seeds.split("-"))
    cfg = json.loads(sys.argv[5]) if len(sys.argv) > 5 else CFG
    _load(src_path)
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(al, s, src_path, cfg) for s in range(a, b + 1) for al in algs
            if not os.path.exists(f"{out_dir}/{al}_{s}.pkl")]
    with Pool(int(os.environ.get("NPROC", 2))) as pool:
        for r in pool.imap_unordered(job, jobs):
            pickle.dump(r, open(f"{out_dir}/{r['alg']}_{r['seed']}.pkl", "wb"))
            Ff = r["F"][r["V"] == 0]
            print(r["alg"], r["seed"], "feas", len(Ff), "best", np.round(Ff.min(0), 3) if len(Ff) else None,
                  "t=%.0fs" % r["time"], r["info"], flush=True)
