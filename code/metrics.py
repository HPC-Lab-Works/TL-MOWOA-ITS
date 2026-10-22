"""HV, IGD, generalized spread and Pareto size on normalized objectives.

Reference front P*: non-dominated union of the final feasible fronts of all
algorithms and runs. Objectives are normalized with the ideal and nadir points
of P*; HV uses reference point (1.1, ..., 1.1)."""
import numpy as np
from pymoo.indicators.hv import HV
from common import nondominated


def reference_front(fronts):
    F = np.vstack([f for f in fronts if len(f)])
    nd = nondominated(F, np.zeros(len(F)))
    Fr = np.unique(F[nd], axis=0)
    return Fr, Fr.min(0), Fr.max(0)


class Scorer:
    def __init__(self, Pstar, ideal, nadir):
        self.sc = np.where(nadir - ideal > 0, nadir - ideal, 1)
        self.ideal = ideal
        self.Pn = (Pstar - ideal) / self.sc
        self.hv = HV(ref_point=np.full(Pstar.shape[1], 1.1))
        m = Pstar.shape[1]
        self.ext = self.Pn[[np.argmax(self.Pn[:, k]) for k in range(m)]]   # extreme solutions

    def norm(self, F):
        return (F - self.ideal) / self.sc

    def HV(self, F):
        if len(F) == 0:
            return 0.0
        return float(self.hv(self.norm(F)))

    def IGD(self, F):
        if len(F) == 0:
            return np.nan
        Fn = self.norm(F)
        d = np.linalg.norm(self.Pn[:, None] - Fn[None], axis=-1).min(1)
        return float(d.mean())

    def spread(self, F):
        """Generalized spread (Zhou et al., 2006); lower is better."""
        if len(F) < 2:
            return np.nan
        S = self.norm(F)
        D = np.linalg.norm(S[:, None] - S[None], axis=-1)
        np.fill_diagonal(D, np.inf)
        dmin = D.min(1)
        dbar = dmin.mean()
        de = np.linalg.norm(self.ext[:, None] - S[None], axis=-1).min(1).sum()
        return float((de + np.abs(dmin - dbar).sum()) / (de + len(S) * dbar))
