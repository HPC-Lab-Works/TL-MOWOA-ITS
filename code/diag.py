"""Diagnostic (tuning seeds only): archive-entry success rate by move type in TL-MOWOA."""
import numpy as np, problem as P, algorithms as A, run as R
from common import to_unit, from_unit, lhs, Archive
R._load('/home/claude/tlmowoa_out/source.pkl')
tot = {}
for seed in (101, 102, 103):
    rng = np.random.default_rng(seed); ev = A.Evaluator(P.TARGET); arch = Archive(A.M)
    tm = R.tm_for(seed)
    Xt = from_unit(lhs(A.N_T, P.NVAR, rng)); Ft, Vt = ev(Xt); arch.update(Xt, Ft, Vt); tm.fine_tune(Xt, Ft, Vt).build_priors()
    U = lhs(A.N, P.NVAR, rng); m = rng.random(A.N) < 0.1
    if m.any(): U[m] = to_unit(tm.sample_prior(m.sum()))
    F, V = ev(from_unit(U)); arch.update(from_unit(U), F, V)
    gens = (A.BUDGET - ev.n) // A.N
    for t in range(gens):
        Lead = to_unit(A.pick_leaders(arch, A.N, rng)); tlm = rng.random(A.N) < 0.1
        Lead[tlm] = to_unit(tm.sample_prior(tlm.sum(), sigma=0.0)) if tlm.any() else Lead[tlm]
        a = 2 * (1 - t / gens); r = rng.random(A.N); Av = 2 * a * r - a; p = rng.random(A.N)
        kind = np.where(np.abs(Av) >= 1, "search", np.where(p < 0.5, "encircle", "spiral"))
        # reproduce woa_move with the same draws
        Cv = 2 * rng.random((A.N, P.NVAR)); l = rng.uniform(-1, 1, A.N); out = np.empty_like(U)
        e, s, q = kind == "encircle", kind == "spiral", kind == "search"
        out[e] = Lead[e] - Av[e, None] * np.abs(Cv[e] * Lead[e] - U[e])
        out[s] = np.abs(Lead - U)[s] * np.exp(l[s, None]) * np.cos(2 * np.pi * l[s, None]) + Lead[s]
        Xr = U[rng.integers(0, A.N, A.N)]; out[q] = Xr[q] - Av[q, None] * np.abs(Cv[q] * Xr[q] - U[q])
        Un = np.clip(out, 0, 1); Xn = from_unit(Un); Fn, Vn = ev(Xn); arch.update(Xn, Fn, Vn); U = Un
        inarch = np.array([np.any(np.all(np.isclose(arch.X, x), 1)) for x in Xn])
        half = "early" if t < gens / 2 else "late"
        for k in ("encircle", "spiral", "search"):
            for src, msk in (("TL-leader", tlm), ("arch-leader", ~tlm)):
                sel = (kind == k) & msk
                key = (k, src, half); n0, n1 = tot.get(key, (0, 0)); tot[key] = (n0 + sel.sum(), n1 + inarch[sel].sum())
for k, (n, s) in sorted(tot.items()):
    print(f"{k[0]:9s} {k[1]:11s} {k[2]:5s} moves {n:5d}  entered archive {s:4d}  rate {s / max(n, 1):.3f}")
