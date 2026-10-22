"""Transfer learning module (Section 4.2 of the manuscript).

1. Source surrogate M_s pre-trained on historical optimization data D_s
   (all solutions evaluated while optimising four source demand scenarios).
2. Target model initialised from theta_s* and fine-tuned on n_t target samples
   with  L = MSE + beta * MMD^2(phi(D_s), phi(D_t)) + gamma * ||theta - theta_s*||^2.
3. Kernel mean matching weights for source candidates.
4. Transfer priors S_prior: top-k source archive solutions ranked by the
   fine-tuned surrogate's predicted constrained non-dominance on the target,
   ties broken by KMM weight.
5. Latent regions R_p: k-means clusters of embeddings of the best candidates.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import problem as P
from common import to_unit, from_unit, fast_sort

torch.set_num_threads(1)
IN_DIM = P.NVAR + P.N_ROUTES


class Surrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(IN_DIM, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 5))

    def forward(self, x):
        z = self.phi(x)
        return self.head(z), z


def make_inputs(X, demand):
    U = to_unit(X)
    d = np.tile(P.demand_features(demand), (len(U), 1))
    return np.hstack([U, d]).astype(np.float32)


class TargetScaler:
    def __init__(self, F, V):
        self.mu = F.mean(0); self.sd = F.std(0) + 1e-9

    def fwd(self, F, V):
        return np.hstack([(F - self.mu) / self.sd, np.log1p(V)[:, None]]).astype(np.float32)

    def inv(self, Y):
        return Y[:, :4] * self.sd + self.mu, np.expm1(np.maximum(Y[:, 4], 0))


def mmd2(a, b, sigma=None):
    x = torch.cat([a, b])
    d = torch.cdist(x, x) ** 2
    if sigma is None:
        sigma = d.detach().median().clamp_min(1e-6)
    K = torch.exp(-d / sigma)
    n = len(a)
    return K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean()


def pretrain_source(src, epochs=40, seed=0):
    torch.manual_seed(seed)
    Xin = np.vstack([make_inputs(s["X_all"], s["demand"]) for s in src])
    F = np.vstack([s["F_all"] for s in src]); V = np.concatenate([s["V_all"] for s in src])
    sc = TargetScaler(F, V)
    Y = sc.fwd(F, V)
    m = Surrogate()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    xt, yt = torch.tensor(Xin), torch.tensor(Y)
    for ep in range(epochs):
        perm = torch.randperm(len(xt))
        for b in range(0, len(xt), 256):
            i = perm[b:b + 256]
            loss = ((m(xt[i])[0] - yt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m, sc, Xin


class TransferModule:
    def __init__(self, src, model_s, scaler, Xin_src, demand_t, beta=0.57, gamma=0.39,
                 k=7, n_regions=3, seed=0):
        self.src, self.ms, self.sc, self.Xs = src, model_s, scaler, Xin_src
        self.dem = demand_t
        self.beta, self.gamma, self.k, self.nr = beta, gamma, k, n_regions
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

    def fine_tune(self, Xt, Ft, Vt, epochs=150):
        m = Surrogate(); m.load_state_dict(self.ms.state_dict())
        theta_s = [p.detach().clone() for p in m.parameters()]
        npar = sum(p.numel() for p in theta_s)
        opt = torch.optim.Adam(m.parameters(), 5e-4)
        xt = torch.tensor(make_inputs(Xt, self.dem)); yt = torch.tensor(self.sc.fwd(Ft, Vt))
        xs = torch.tensor(self.Xs)
        for ep in range(epochs):
            j = torch.randint(0, len(xs), (len(xt),))
            pred, zt = m(xt)
            _, zs = m(xs[j])
            reg = sum(((p - q) ** 2).sum() for p, q in zip(m.parameters(), theta_s)) / npar
            loss = ((pred - yt) ** 2).mean() + self.beta * mmd2(zs, zt) + self.gamma * reg
            opt.zero_grad(); loss.backward(); opt.step()
        self.mt = m.eval()
        with torch.no_grad():
            self.zt = m(xt)[1].numpy()
        return self

    def predict(self, X):
        with torch.no_grad():
            Y, Z = self.mt(torch.tensor(make_inputs(X, self.dem)))
        F, V = self.sc.inv(Y.numpy())
        return F, V, Z.numpy()

    def kmm_weights(self, Zs, B=10.0, eps=0.1, iters=300):
        """Kernel mean matching by projected gradient descent."""
        Zt = self.zt
        allz = np.vstack([Zs, Zt])
        s2 = np.median(((allz[:200, None] - allz[None, :200]) ** 2).sum(-1)) + 1e-9
        K = np.exp(-((Zs[:, None] - Zs[None]) ** 2).sum(-1) / s2)
        kap = np.exp(-((Zs[:, None] - Zt[None]) ** 2).sum(-1) / s2).mean(1)
        ns = len(Zs)
        w = np.ones(ns)
        lr = 1.0 / (np.linalg.norm(K, 2) / ns ** 2 + 1e-9)
        for _ in range(iters):
            grad = (K @ w) / ns ** 2 - kap / ns
            w = np.clip(w - lr * grad, 0, B)
            w *= np.clip(np.mean(w), 1 - eps, 1 + eps) / max(np.mean(w), 1e-9)
        return w

    def build_priors(self):
        cand = np.unique(np.vstack([s["X_arch"] for s in self.src]), axis=0)
        Fp, Vp, Z = self.predict(cand)
        Vp = np.where(Vp < 0.05, 0.0, Vp)
        rank = fast_sort(Fp, Vp)
        w = self.kmm_weights(Z)
        order = np.lexsort((-w, rank))
        self.priors = cand[order[:self.k]]
        top = order[:max(self.k, len(order) // 5)]
        km = KMeans(self.nr, n_init=5, random_state=0).fit(Z[top])
        self.reg_members = [cand[top][km.labels_ == p] for p in range(self.nr)]
        self.mu = km.cluster_centers_
        self.r = np.array([np.percentile(np.linalg.norm(Z[top][km.labels_ == p] - self.mu[p], axis=1), 80)
                           if (km.labels_ == p).sum() else 0.0 for p in range(self.nr)])
        self.pred_rank0 = int((rank == 0).sum())
        return self

    def sample_prior(self, n, sigma=0.05):
        """Mixture of transfer priors and latent-region samples (p_prior)."""
        out = []
        for _ in range(n):
            if self.rng.random() < 0.5:
                base = self.priors[self.rng.integers(len(self.priors))]
            else:
                p = self.rng.integers(self.nr)
                mem = self.reg_members[p]
                base = mem[self.rng.integers(len(mem))]
            u = to_unit(base[None])[0] + self.rng.normal(0, sigma, P.NVAR)
            out.append(from_unit(u))
        return np.array(out)

    def in_region(self, X):
        _, _, Z = self.predict(X)
        d = np.linalg.norm(Z[:, None] - self.mu[None], axis=-1)
        return (d <= self.r[None]).any(1)
