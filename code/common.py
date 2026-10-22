"""Shared utilities: sampling, constrained dominance, crowding, archive."""
import numpy as np
import problem as P


def to_unit(X):
    return (X - P.LB) / (P.UB - P.LB)


def from_unit(U):
    return P.LB + np.clip(U, 0, 1) * (P.UB - P.LB)


def lhs(n, d, rng):
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.random((n, d))) / n
    return u


def cdom_matrix(F, V):
    """D[i,j] = True if i constraint-dominates j (Deb's rules)."""
    Vi, Vj = V[:, None], V[None, :]
    le = (F[:, None, :] <= F[None, :, :]).all(-1)
    lt = (F[:, None, :] < F[None, :, :]).any(-1)
    pareto = le & lt
    both_feas = (Vi == 0) & (Vj == 0)
    return np.where(both_feas, pareto, Vi < Vj)


def nondominated(F, V):
    D = cdom_matrix(F, V)
    return ~D.any(0)


def fast_sort(F, V):
    D = cdom_matrix(F, V)
    n = len(F)
    cnt = D.sum(0)
    rank = np.full(n, -1)
    r = 0
    cur = np.where(cnt == 0)[0]
    while len(cur):
        rank[cur] = r
        cnt = cnt - D[cur].sum(0)
        cnt[rank >= 0] = -1
        cur = np.where(cnt == 0)[0]
        r += 1
    return rank


def crowding(F):
    n, m = F.shape
    if n <= 2:
        return np.full(n, np.inf)
    cd = np.zeros(n)
    rng_ = F.max(0) - F.min(0)
    rng_[rng_ == 0] = 1
    for k in range(m):
        o = np.argsort(F[:, k])
        cd[o[0]] = cd[o[-1]] = np.inf
        cd[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / rng_[k]
    return cd


class Archive:
    def __init__(self, M):
        self.M = M
        self.X = np.empty((0, P.NVAR))
        self.F = np.empty((0, 4))
        self.V = np.empty(0)

    def update(self, X, F, V):
        X = np.vstack([self.X, X]); F = np.vstack([self.F, F]); V = np.concatenate([self.V, V])
        # remove exact duplicates
        _, u = np.unique(np.round(X, 9), axis=0, return_index=True)
        X, F, V = X[u], F[u], V[u]
        if (V == 0).any():
            nd = nondominated(F, V)
        else:   # no feasible solution yet: keep the M least-violated solutions
            nd = np.argsort(V)[: self.M]
        X, F, V = X[nd], F[nd], V[nd]
        while len(X) > self.M:
            cd = crowding(F)
            j = np.argmin(cd)
            X, F, V = np.delete(X, j, 0), np.delete(F, j, 0), np.delete(V, j)
        self.X, self.F, self.V = X, F, V

    def feasible_F(self):
        return self.F[self.V == 0]


class Recorder:
    """Records, after every generation, the number of true evaluations, the
    feasible non-dominated set and the best feasible value of each objective."""
    def __init__(self):
        self.evals, self.fronts, self.best = [], [], []

    def log(self, evals, Ffeas):
        self.evals.append(evals)
        self.fronts.append(Ffeas.copy())
        self.best.append(Ffeas.min(0) if len(Ffeas) else np.full(4, np.nan))
