"""TL-MOWOA and baselines. Every algorithm is given the same budget of
true target evaluations (BUDGET) and maintains the same external archive (M)."""
import numpy as np
import torch
import torch.nn as nn
import problem as P
from common import to_unit, from_unit, lhs, Archive, Recorder, crowding, fast_sort, nondominated

N, T, M, BUDGET = 100, 50, 100, 5000
N_T = 200            # target samples used for fine-tuning (counted in the budget)
B_SPIRAL = 1.0
torch.set_num_threads(1)


class Evaluator:
    def __init__(self, demand, keep=False):
        self.dem, self.n, self.keep, self.store = demand, 0, keep, []

    def __call__(self, X):
        F, G = P.evaluate(X, self.dem)
        self.n += len(X)
        V = P.violation(G)
        if self.keep:
            self.store.append((X.copy(), F, V))
        return F, V


def pick_leaders(arch, n, rng):
    """Crowding-distance-based roulette over the archive (less crowded = likelier)."""
    cd = crowding(arch.F)
    fin = np.isfinite(cd)
    cd = np.where(fin, cd, (cd[fin].max() if fin.any() else 1.0) * 2 + 1e-9)
    p = cd / cd.sum()
    return arch.X[rng.choice(len(arch.X), n, p=p)]


def woa_move(U, Lead, t, Tmax, rng, a_max=2.0, Xref=None):
    """Standard WOA update. a decreases linearly from a_max to 0. If Xref is
    given, search-for-prey uses Xref (e.g. transfer samples) as the reference
    agent instead of a random population member."""
    n, d = U.shape
    a = a_max * (1 - t / Tmax)
    r = rng.random(n)
    A = 2 * a * r - a
    Cv = 2 * rng.random((n, d))
    p = rng.random(n)
    l = rng.uniform(-1, 1, n)
    out = np.empty_like(U)
    enc = (np.abs(A) < 1) & (p < 0.5)
    spi = (np.abs(A) < 1) & (p >= 0.5)
    sea = np.abs(A) >= 1
    out[enc] = Lead[enc] - A[enc, None] * np.abs(Cv[enc] * Lead[enc] - U[enc])
    Dp = np.abs(Lead - U)
    out[spi] = (Dp[spi] * np.exp(B_SPIRAL * l[spi, None]) * np.cos(2 * np.pi * l[spi, None]) + Lead[spi])
    Xr = U[rng.integers(0, n, n)] if Xref is None else Xref
    out[sea] = Xr[sea] - A[sea, None] * np.abs(Cv[sea] * Xr[sea] - U[sea])
    return np.clip(out, 0, 1)


def controller_indicators(Upop, Fpop, arch):
    Ff = arch.feasible_F()
    ref = Ff if len(Ff) else arch.F
    lo, hi = ref.min(0), ref.max(0)
    sc = np.where(hi - lo > 0, hi - lo, 1)
    Fn = (ref - lo) / sc
    best = ref[np.argmin(np.linalg.norm(Fn, axis=1))]
    Ct = np.linalg.norm((Fpop - best) / sc, axis=1).mean()
    diff = np.linalg.norm(Upop[:, None] - Upop[None], axis=-1)
    Dt = diff.sum() / (len(Upop) * (len(Upop) - 1)) / np.sqrt(Upop.shape[1])
    return Ct, Dt


def run_woa_family(demand, seed, tm=None, tl_init=False, tl_leader=False, controller=False,
                   p_tl=0.3, tau_c=0.5, tau_d=0.12, sigma_loc=0.02, keep=False,
                   a_max=2.0, tl_search=False, adaptive_ptl=False, eta_ptl=0.2,
                   ptl_bounds=(0.02, 0.5), loc_decay=False):
    """MOWOA (no TL), Hybrid (TL init only) and TL-MOWOA (TL init + TL leaders
    + adaptive controller) share this implementation."""
    rng = np.random.default_rng(seed)
    ev = Evaluator(demand, keep)
    arch, rec = Archive(M), Recorder()
    info = {"exploit_gens": 0, "ev": ev}
    if tm is not None:   # target samples for fine-tuning (counted)
        Xt = from_unit(lhs(N_T, P.NVAR, rng))
        Ft, Vt = ev(Xt)
        arch.update(Xt, Ft, Vt)
        tm.fine_tune(Xt, Ft, Vt).build_priors()
        rec.log(ev.n, arch.feasible_F())
    if tl_init:
        mask = rng.random(N) < p_tl
        U = lhs(N, P.NVAR, rng)
        if mask.any():
            U[mask] = to_unit(tm.sample_prior(mask.sum()))
    else:
        U = lhs(N, P.NVAR, rng)
    X = from_unit(U)
    F, V = ev(X)
    arch.update(X, F, V)
    rec.log(ev.n, arch.feasible_F())
    gens = (BUDGET - ev.n) // N
    phase_exploit = False
    info["p_tl_trace"] = []
    for t in range(gens):
        Lead = to_unit(pick_leaders(arch, N, rng))
        m = np.zeros(N, bool)
        if tl_leader:
            m = rng.random(N) < p_tl
            if m.any():
                Lead[m] = to_unit(tm.sample_prior(m.sum(), sigma=0.0))
        if phase_exploit:
            s = sigma_loc * (0.1 ** (t / gens)) if loc_decay else sigma_loc
            Unew = np.clip(to_unit(pick_leaders(arch, N, rng)) + rng.normal(0, s, U.shape), 0, 1)
            info["exploit_gens"] += 1
        else:
            Xref = to_unit(tm.sample_prior(N, sigma=0.05)) if tl_search else None
            Unew = woa_move(U, Lead, t, gens, rng, a_max=a_max, Xref=Xref)
        X = from_unit(Unew)
        F, V = ev(X)
        U = Unew
        arch.update(X, F, V)
        if adaptive_ptl and tl_leader:
            keys = {tuple(np.round(x, 9)) for x in arch.X}
            entered = np.array([tuple(np.round(x, 9)) in keys for x in X])
            s_tl = entered[m].mean() if m.any() else 0.0
            s_ar = entered[~m].mean() if (~m).any() else 0.0
            p_tl = float(np.clip(p_tl * (1 + eta_ptl) if s_tl > s_ar else p_tl * (1 - eta_ptl), *ptl_bounds))
            info["p_tl_trace"].append(p_tl)
        rec.log(ev.n, arch.feasible_F())
        if controller:
            Ct, Dt = controller_indicators(U, F, arch)
            phase_exploit = (Ct > tau_c) and (Dt < tau_d)
    assert ev.n <= BUDGET
    return arch, rec, info


def run_tl_only(demand, seed, tm):
    """Transfer-only search: candidates are drawn from the transfer priors and
    latent regions, or as Gaussian perturbations of archive members, with a
    decreasing step size; no evolutionary operators."""
    rng = np.random.default_rng(seed)
    ev = Evaluator(demand)
    arch, rec = Archive(M), Recorder()
    Xt = from_unit(lhs(N_T, P.NVAR, rng))
    Ft, Vt = ev(Xt)
    arch.update(Xt, Ft, Vt)
    tm.fine_tune(Xt, Ft, Vt).build_priors()
    rec.log(ev.n, arch.feasible_F())
    gens = (BUDGET - ev.n) // N
    for t in range(gens):
        s = 0.1 * (0.01 / 0.1) ** (t / max(gens - 1, 1))
        half = N // 2
        Xa = tm.sample_prior(half, sigma=s)
        Ua = np.clip(to_unit(pick_leaders(arch, N - half, rng)) + rng.normal(0, s, (N - half, P.NVAR)), 0, 1)
        X = np.vstack([Xa, from_unit(Ua)])
        F, V = ev(X)
        arch.update(X, F, V)
        rec.log(ev.n, arch.feasible_F())
    return arch, rec, {}


# ---------------------------- variation operators ----------------------------
def sbx(P1, P2, rng, eta=30):
    """pymoo SBX (pymoo NSGA-III defaults: eta=30, prob_var=0.5); returns both children."""
    from pymoo.operators.crossover.sbx import cross_sbx
    n, d = P1.shape
    Q = cross_sbx(np.stack([P1, P2]), np.zeros(d), np.ones(d), np.full((n, 1), float(eta)),
                  np.full((n, 1), 0.5), np.full((n, 1), 0.5), random_state=rng)
    return np.clip(np.vstack([Q[0], Q[1]]), 0, 1)


def pm(U, rng, eta=20):
    """pymoo polynomial mutation, probability 1/n per variable."""
    from pymoo.operators.mutation.pm import mut_pm
    n, d = U.shape
    out = mut_pm(U.copy(), np.zeros(d), np.ones(d), np.full(n, float(eta)), np.full(n, 1.0 / d), True,
                 random_state=rng)
    return np.clip(out, 0, 1)


def nsga2_select(U, F, V, n):
    rank = fast_sort(F, V)
    sel = []
    for r in range(rank.max() + 1):
        idx = np.where(rank == r)[0]
        if len(sel) + len(idx) <= n:
            sel += list(idx)
        else:
            cd = crowding(F[idx])
            sel += list(idx[np.argsort(-cd)[: n - len(sel)]])
            break
    return np.array(sel)


# ------------------------------- NSGA-III (pymoo) ------------------------------
def run_nsga3(demand, seed):
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.core.callback import Callback

    ev = Evaluator(demand)
    arch, rec = Archive(M), Recorder()

    class Prob(Problem):
        def __init__(self):
            super().__init__(n_var=P.NVAR, n_obj=4, n_ieq_constr=3, xl=P.LB, xu=P.UB)

        def _evaluate(self, X, out, *a, **k):
            F, G = P.evaluate(X, demand)
            ev.n += len(X)
            out["F"], out["G"] = F, G
            arch.update(X, F, P.violation(G))

    class CB(Callback):
        def notify(self, algorithm):
            rec.log(ev.n, arch.feasible_F())

    ref = get_reference_directions("das-dennis", 4, n_partitions=5)
    alg = NSGA3(ref_dirs=ref, pop_size=N)
    minimize(Prob(), alg, ("n_eval", BUDGET), seed=seed, callback=CB(), verbose=False)
    return arch, rec, {}


# ------------------------- GCN-surrogate-assisted EA -------------------------
def _adj():
    A = np.eye(P.NI)
    for r in range(P.R):
        for c in range(P.C_GRID):
            i = P.idx(r, c)
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr < P.R and cc < P.C_GRID:
                    j = P.idx(rr, cc); A[i, j] = A[j, i] = 1
    d = A.sum(1) ** -0.5
    return torch.tensor(d[:, None] * A * d[None], dtype=torch.float32)


A_HAT = _adj()


def _node_feats(U, demand):
    """Node features: [cycle, split, offset] (unit scale), the four route
    demands through the node, and a one-hot positional encoding."""
    n = len(U)
    node = U.reshape(n, P.NI, 3)
    dem = P.demand_features(demand)
    nd = np.zeros((P.NI, 4))
    for r in range(P.R):
        for c in range(P.C_GRID):
            i = P.idx(r, c)
            nd[i] = [dem[2 * r], dem[2 * r + 1], dem[2 * P.R + 2 * c], dem[2 * P.R + 2 * c + 1]]
    pe = np.eye(P.NI)
    return np.concatenate([node, np.broadcast_to(nd, (n, P.NI, 4)), np.broadcast_to(pe, (n, P.NI, P.NI))],
                          -1).astype(np.float32)


class GCN(nn.Module):
    """Three residual graph-convolution layers, mean and max readout, MLP head."""
    def __init__(self, h=64):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(7 + P.NI, h), nn.Linear(h, h), nn.Linear(h, h)
        self.out = nn.Sequential(nn.Linear(2 * h, 64), nn.ReLU(), nn.Linear(64, 5))

    def forward(self, x):
        h = torch.relu(A_HAT @ self.l1(x))
        h = torch.relu(A_HAT @ self.l2(h)) + h
        h = torch.relu(A_HAT @ self.l3(h)) + h
        return self.out(torch.cat([h.mean(1), h.max(1).values], -1))


def _fit_gcn(Ud, F, V, demand, seed, epochs=120):
    """Targets: log objectives and log(1+violation), standardised."""
    torch.manual_seed(seed)
    Yr = np.hstack([np.log(F), np.log1p(V)[:, None]])
    mu, sd = Yr.mean(0), Yr.std(0) + 1e-9
    Y = torch.tensor((Yr - mu) / sd, dtype=torch.float32)
    Xn = torch.tensor(_node_feats(Ud, demand))
    m = GCN()
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(Xn))
        for b in range(0, len(Xn), 128):
            i = perm[b:b + 128]
            loss = ((m(Xn[i]) - Y[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval(), mu, sd


def run_gcn_saea(demand, seed, n_cand=5, retrain_every=5):
    """NSGA-II-style EA in which a graph convolutional surrogate pre-screens
    n_cand*N offspring per generation; only the N most promising (by predicted
    constrained non-dominated sorting and crowding) are truly evaluated."""
    rng = np.random.default_rng(seed)
    ev = Evaluator(demand)
    arch, rec = Archive(M), Recorder()
    U = lhs(N, P.NVAR, rng)
    F, V = ev(from_unit(U))
    arch.update(from_unit(U), F, V)
    rec.log(ev.n, arch.feasible_F())
    DU, DF, DV = U.copy(), F.copy(), V.copy()
    model = None
    gens = (BUDGET - ev.n) // N
    for t in range(gens):
        if t % retrain_every == 0:
            model = _fit_gcn(DU[-1500:], DF[-1500:], DV[-1500:], demand, seed + t, epochs=60)
        m_, mu, sd = model
        k = n_cand * N // 2          # each mating yields two children
        i1, i2 = rng.integers(0, N, k), rng.integers(0, N, k)
        # binary tournament on rank
        rank = fast_sort(F, V)
        j1, j2 = rng.integers(0, N, k), rng.integers(0, N, k)
        par1 = np.where(rank[i1] <= rank[j1], i1, j1)
        par2 = np.where(rank[i2] <= rank[j2], i2, j2)
        Cand = pm(sbx(U[par1], U[par2], rng), rng)
        with torch.no_grad():
            Yp = m_(torch.tensor(_node_feats(Cand, demand))).numpy()
        Yr = Yp * sd + mu
        Fp = np.exp(Yr[:, :4])
        Vp = np.expm1(np.maximum(Yr[:, 4], 0)); Vp = np.where(Vp < 0.05, 0, Vp)
        pick = nsga2_select(Cand, Fp, Vp, N)
        Uo = Cand[pick]
        Fo, Vo = ev(from_unit(Uo))
        arch.update(from_unit(Uo), Fo, Vo)
        DU, DF, DV = np.vstack([DU, Uo]), np.vstack([DF, Fo]), np.concatenate([DV, Vo])
        Uall, Fall, Vall = np.vstack([U, Uo]), np.vstack([F, Fo]), np.concatenate([V, Vo])
        s = nsga2_select(Uall, Fall, Vall, N)
        U, F, V = Uall[s], Fall[s], Vall[s]
        rec.log(ev.n, arch.feasible_F())
    return arch, rec, {}
