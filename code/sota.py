"""State-of-the-art comparators added after the IET ITS review.

C-TAEA, AGE-MOEA-II and CMOPSO use their pymoo 0.6.2 implementations; CCMO is
implemented here. Every algorithm receives the same budget of true target
evaluations (BUDGET) and is scored on the same external archive (M) as the
other methods in algorithms.py."""
import numpy as np
import problem as P
from common import from_unit, lhs, Archive, Recorder, cdom_matrix
from algorithms import N, M, BUDGET, Evaluator, sbx, pm


# ------------------------------ pymoo wrapper ------------------------------
def run_pymoo(demand, seed, name):
    from pymoo.core.problem import Problem
    from pymoo.core.callback import Callback
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    ev = Evaluator(demand)
    arch, rec = Archive(M), Recorder()

    class Prob(Problem):
        def __init__(self):
            super().__init__(n_var=P.NVAR, n_obj=4, n_ieq_constr=3, xl=P.LB, xu=P.UB)

        def _evaluate(self, X, out, *a, **k):
            F, G = P.evaluate(X, demand)
            out["F"], out["G"] = F, G
            room = max(BUDGET - ev.n, 0)          # never count evaluations beyond the budget
            if room:
                arch.update(X[:room], F[:room], P.violation(G[:room]))
            ev.n += len(X)

    class CB(Callback):
        def notify(self, algorithm):
            rec.log(min(ev.n, BUDGET), arch.feasible_F())

    if name == "C-TAEA":
        from pymoo.algorithms.moo.ctaea import CTAEA
        from pymoo.util.ref_dirs import get_reference_directions
        alg = CTAEA(ref_dirs=get_reference_directions("energy", 4, N, seed=1))
    elif name == "AGE-MOEA-II":
        from pymoo.algorithms.moo.age2 import AGEMOEA2
        alg = AGEMOEA2(pop_size=N)
    elif name == "CMOPSO":
        from pymoo.algorithms.moo.cmopso import CMOPSO
        alg = CMOPSO(pop_size=N)
    else:
        raise ValueError(name)
    minimize(Prob(), alg, get_termination("n_eval", BUDGET), seed=seed, callback=CB(), verbose=False)
    return arch, rec, {"evals_used": min(ev.n, BUDGET)}


# ----------------------------------- CCMO -----------------------------------
def _spea2_fitness(F, V, constrained):
    """SPEA2 fitness (raw fitness + density). With constrained=True dominance
    follows Deb's constrained-dominance principle; otherwise constraints are
    ignored (helper problem)."""
    n = len(F)
    D = cdom_matrix(F, V if constrained else np.zeros(n))
    S = D.sum(1)
    R = (D * S[:, None]).sum(0)                    # sum of strengths of dominators
    dist = np.linalg.norm(F[:, None] - F[None], axis=-1)
    np.fill_diagonal(dist, np.inf)
    k = int(np.floor(np.sqrt(n)))
    sk = np.sort(dist, 1)[:, k - 1]
    return R + 1.0 / (sk + 2.0)


def _spea2_select(F, V, n, constrained):
    fit = _spea2_fitness(F, V, constrained)
    nxt = np.where(fit < 1)[0]
    if len(nxt) < n:
        nxt = np.argsort(fit)[:n]
    elif len(nxt) > n:                              # SPEA2 truncation by nearest distances
        nxt = list(nxt)
        Fs = F[nxt]
        dist = np.linalg.norm(Fs[:, None] - Fs[None], axis=-1)
        np.fill_diagonal(dist, np.inf)
        alive = np.ones(len(nxt), bool)
        while alive.sum() > n:
            idx = np.where(alive)[0]
            ds = np.sort(dist[np.ix_(idx, idx)], 1)
            order = np.lexsort(ds.T[::-1])          # smallest distance, ties by next distance
            alive[idx[order[0]]] = False
        nxt = np.array(nxt)[alive]
    return np.asarray(nxt), fit


def _tournament(fit, n, rng):
    a, b = rng.integers(0, len(fit), n), rng.integers(0, len(fit), n)
    return np.where(fit[a] < fit[b], a, b)


def run_ccmo(demand, seed):
    """Coevolutionary constrained multi-objective optimization (CCMO): Pop1
    solves the constrained problem and Pop2 the problem without constraints;
    each generation N/2 offspring are produced from each population (SBX and
    polynomial mutation), and both populations select N survivors from their
    own members and both offspring sets with SPEA2 environmental selection."""
    rng = np.random.default_rng(seed)
    ev = Evaluator(demand)
    arch, rec = Archive(M), Recorder()
    pops = []
    for _ in range(2):
        U = lhs(N, P.NVAR, rng)
        F, V = ev(from_unit(U))
        arch.update(from_unit(U), F, V)
        pops.append([U, F, V])
    rec.log(ev.n, arch.feasible_F())
    fits = [_spea2_fitness(pops[0][1], pops[0][2], True), _spea2_fitness(pops[1][1], pops[1][2], False)]
    half = N // 2
    while ev.n + N <= BUDGET:
        offs = []
        for (U, F, V), fit in zip(pops, fits):
            mp = _tournament(fit, N, rng)
            par = U[mp]
            # N parents form N/2 pairs; the first child of each pair is kept (N/2 offspring)
            Q = pm(sbx(par[:half], par[half:2 * half], rng, eta=20)[:half], rng)
            Fq, Vq = ev(from_unit(Q))
            arch.update(from_unit(Q), Fq, Vq)
            offs.append((Q, Fq, Vq))
        Uo = np.vstack([o[0] for o in offs]); Fo = np.vstack([o[1] for o in offs]); Vo = np.concatenate([o[2] for o in offs])
        new = []
        for j, (U, F, V) in enumerate(pops):
            Ua, Fa, Va = np.vstack([U, Uo]), np.vstack([F, Fo]), np.concatenate([V, Vo])
            sel, _ = _spea2_select(Fa, Va, N, constrained=(j == 0))
            new.append([Ua[sel], Fa[sel], Va[sel]])
        pops = new
        fits = [_spea2_fitness(pops[0][1], pops[0][2], True), _spea2_fitness(pops[1][1], pops[1][2], False)]
        rec.log(ev.n, arch.feasible_F())
    assert ev.n <= BUDGET
    return arch, rec, {}
