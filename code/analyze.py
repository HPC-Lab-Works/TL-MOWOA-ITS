"""Aggregate the 30-seed results: metrics, statistics and figures."""
import sys, glob, pickle, json, numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import metrics as Mx

RUNS, OUT = sys.argv[1], sys.argv[2]
ALGS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["TL-MOWOA", "MOWOA", "NSGA-III", "TL", "Hybrid", "GNN"]
REF = ALGS[0]          # algorithm against which the Wilcoxon tests are made
PLOT = sys.argv[4].split(",") if len(sys.argv) > 4 else ALGS   # algorithms shown in figures
LABEL = {"TL-MOWOA": "TL-MOWOA ($a_{max}$=2, original)", "TL-MOWOA-v2": "TL-MOWOA", "MOWOA": "MOWOA",
         "MOWOA-a1": "MOWOA ($a_{max}$=1)", "NSGA-III": "NSGA-III", "TL": "TL algorithm",
         "Hybrid": "Hybrid (TL init + MOWOA)", "Hybrid-a1": "Hybrid ($a_{max}$=1)", "GNN": "GCN-surrogate EA"}
SHORT = {"TL-MOWOA": "TL-MOWOA\na=2", "TL-MOWOA-v2": "TL-MOWOA", "MOWOA": "MOWOA", "MOWOA-a1": "MOWOA\na=1",
         "NSGA-III": "NSGA-III", "TL": "TL", "Hybrid": "Hybrid", "Hybrid-a1": "Hybrid\na=1", "GNN": "GCN-EA"}
COL = {"TL-MOWOA-v2": "#1F5FAD", "TL-MOWOA": "#6FA8DC", "MOWOA": "#E8833A", "MOWOA-a1": "#B8860B",
       "NSGA-III": "#3A9D5D", "TL": "#C8435A", "Hybrid": "#8B6CC7", "Hybrid-a1": "#5E3C99", "GNN": "#7A7A7A"}
R = {a: sorted([pickle.load(open(f, "rb")) for f in glob.glob(f"{RUNS}/{a}_*.pkl")], key=lambda r: r["seed"])
     for a in ALGS}
n_runs = {a: len(R[a]) for a in ALGS}
final = {a: [r["F"][r["V"] == 0] for r in R[a]] for a in ALGS}
Ps, lo, hi = Mx.reference_front([f for a in ALGS for f in final[a]])
S = Mx.Scorer(Ps, lo, hi)

res = {}
for a in ALGS:
    hv = np.array([S.HV(f) for f in final[a]])
    igd = np.array([S.IGD(f) for f in final[a]])
    sp = np.array([S.spread(f) for f in final[a]])
    ps = np.array([len(f) for f in final[a]])
    first = []
    for r in R[a]:
        e = [ev for ev, fr in zip(r["evals"], r["fronts"]) if len(fr)]
        first.append(e[0] if e else np.nan)
    best = np.array([f.min(0) if len(f) else np.full(4, np.nan) for f in final[a]])
    res[a] = dict(hv=hv, igd=igd, sp=sp, ps=ps, feas=np.array([len(f) > 0 for f in final[a]]),
                  first=np.array(first, float), best=best, time=np.array([r["time"] for r in R[a]]))


def stat(x):
    x = x[~np.isnan(x)]
    return (x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0, len(x))


def test(a, key, better):
    x, y = res[REF][key], res[a][key]
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(y) < 2:
        return np.nan, "n/a"
    p = mannwhitneyu(x, y, alternative="two-sided").pvalue
    if p >= 0.05:
        return p, "≈"
    tl_better = (np.median(x) > np.median(y)) if better == "max" else (np.median(x) < np.median(y))
    return p, "+" if tl_better else "−"


summary = {"reference_front_size": len(Ps), "ideal": lo.tolist(), "nadir": hi.tolist(), "algs": {}}
for a in ALGS:
    d = {}
    for key, better in (("hv", "max"), ("igd", "min"), ("sp", "min"), ("ps", "max")):
        m, s, n = stat(res[a][key].astype(float))
        p, sym = test(a, key, better) if a != REF else (np.nan, "")
        d[key] = {"mean": m, "std": s, "n": n, "p": p, "sym": sym}
    d["feasible_runs"] = int(res[a]["feas"].sum())
    d["n_runs"] = n_runs[a]
    d["first_feasible_evals"] = stat(res[a]["first"])[:2]
    bm = np.nanmean(res[a]["best"], 0) if res[a]["feas"].any() else [np.nan] * 4
    d["best_obj_mean"] = list(map(float, bm))
    d["time_s"] = float(res[a]["time"].mean())
    if a.startswith("TL-MOWOA"):
        d["exploit_gens"] = stat(np.array([r["info"]["exploit_gens"] for r in R[a]], float))[:2]
    summary["algs"][a] = d
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=float)

# -------- per-generation curves on a common evaluation grid --------
grid = np.arange(300, 5001, 100)


def on_grid(r, vals):
    ev = np.array(r["evals"])
    out = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        j = np.searchsorted(ev, g, side="right") - 1
        if j >= 0:
            out[i] = vals[j]
    return out


curves = {}
for a in ALGS:
    hv_c, igd_c, best_c = [], [], []
    for r in R[a]:
        hv_c.append(on_grid(r, [S.HV(f) for f in r["fronts"]]))
        igd_c.append(on_grid(r, [S.IGD(f) for f in r["fronts"]]))
        best_c.append(np.stack([on_grid(r, r["best"][:, k]) for k in range(4)], 1))
    curves[a] = dict(hv=np.array(hv_c), igd=np.array(igd_c), best=np.array(best_c))
pickle.dump({"grid": grid, "curves": curves, "Ps": Ps, "lo": lo, "hi": hi}, open(f"{OUT}/curves.pkl", "wb"))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
names = ["Average travel time (min)", "Congestion index (mean degree of saturation)",
         "CO$_2$ emissions (g/km)", "Fuel consumption (L/100 km)"]
fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
for k, ax in enumerate(axs.flat):
    for a in PLOT:
        Y = curves[a]["best"][:, :, k]
        ok = np.isfinite(Y).sum(0) >= Y.shape[0] / 2      # plot only where >= half the runs are feasible
        with np.errstate(all="ignore"):
            med = np.where(ok, np.nanmedian(np.where(np.isfinite(Y), Y, np.nan), 0) if np.isfinite(Y).any() else np.nan, np.nan)
            q1 = np.where(ok, np.nanpercentile(Y, 25, 0), np.nan); q3 = np.where(ok, np.nanpercentile(Y, 75, 0), np.nan)
        if not ok.any():
            continue
        ax.plot(grid, med, color=COL[a], lw=2.2 if a == REF else 1.4, label=LABEL[a])
        ax.fill_between(grid, q1, q3, color=COL[a], alpha=0.12, lw=0)
    ax.set_title(names[k]); ax.set_xlabel("True function evaluations"); ax.grid(color="#E5E7EB")
axs[0, 0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_convergence.png", dpi=200)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for ax, key, t in ((axs[0], "hv", "Hypervolume (higher is better)"), (axs[1], "igd", "IGD (lower is better)")):
    for a in PLOT:
        Y = curves[a][key]
        if key == "hv":
            if np.nanmax(np.nanmedian(Y, 0)) == 0:
                continue
            med = np.nanmedian(Y, 0)
        else:
            med = np.array([np.nanmedian(c) if np.isfinite(c).sum() >= len(c) / 2 else np.nan for c in Y.T])
        ax.plot(grid, med, color=COL[a], lw=2.2 if a == REF else 1.4, label=LABEL[a])
    ax.set_title(t); ax.set_xlabel("True function evaluations"); ax.grid(color="#E5E7EB")
axs[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_hv_igd.png", dpi=200)

# -------- final metrics bar chart (mean ± std) --------
fig, axs = plt.subplots(2, 2, figsize=(11, 7))
for ax, (key, t) in zip(axs.flat, (("hv", "Hypervolume (higher is better)"), ("igd", "IGD (lower is better)"),
                                   ("sp", "Generalized spread (lower is better)"), ("ps", "Pareto set size"))):
    m = [summary["algs"][a][key]["mean"] for a in PLOT]
    s = [summary["algs"][a][key]["std"] for a in PLOT]
    b = ax.bar(range(len(PLOT)), np.nan_to_num(m), yerr=np.nan_to_num(s), color=[COL[a] for a in PLOT],
               width=0.65, capsize=3, edgecolor="white", linewidth=1.5)
    ax.set_xticks(range(len(PLOT)), [SHORT[a] for a in PLOT], fontsize=8)
    ax.set_title(t); ax.grid(axis="y", color="#E5E7EB"); ax.set_axisbelow(True)
    for i, (mm, ss) in enumerate(zip(m, s)):
        if np.isfinite(mm):
            ax.text(i, mm + (ss if np.isfinite(ss) else 0) + max(np.nan_to_num(m)) * 0.02,
                    f"{mm:.3f}" if key != "ps" else f"{mm:.1f}", ha="center", fontsize=8, color="#374151")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_bars.png", dpi=200)

# -------- Pareto projections of the median-HV run --------
fig = plt.figure(figsize=(12, 4.8))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
for a in PLOT:
    hv = res[a]["hv"]
    if not res[a]["feas"].any():
        continue
    i = np.argsort(hv)[len(hv) // 2]
    f = final[a][i]
    if len(f) == 0:      # median run has no feasible solution
        continue
    ax1.scatter(f[:, 0], f[:, 1], s=14, color=COL[a], label=LABEL[a], alpha=0.85, edgecolors="white", linewidths=0.3)
    ax2.scatter(f[:, 0], f[:, 1], f[:, 2], s=10, color=COL[a], alpha=0.85)
ax1.set_xlabel("Average travel time (min)"); ax1.set_ylabel("Congestion index"); ax1.grid(color="#E5E7EB")
ax1.legend(fontsize=8, frameon=False)
ax2.set_xlabel("Travel time (min)"); ax2.set_ylabel("Congestion index"); ax2.set_zlabel("CO$_2$ (g/km)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_pareto.png", dpi=200)
print(json.dumps(summary, indent=1, default=float))
