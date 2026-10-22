"""Generate LaTeX tables directly from the experiment outputs, so that every
number in the manuscript is traceable to summary.json and sumo*.json."""
import sys, json, numpy as np
from scipy.stats import mannwhitneyu

summ = json.load(open(sys.argv[1]))
sumo = json.load(open(sys.argv[2]))
out_metrics, out_sumo = sys.argv[3], sys.argv[4]

ORDER = [("TL-MOWOA-v2", "\\textbf{TL-MOWOA (proposed)}"), ("TL-MOWOA", "TL-MOWOA, $a_{max}=2$"),
         ("MOWOA", "MOWOA"), ("MOWOA-a1", "MOWOA, $a_{max}=1$"), ("Hybrid", "Hybrid"),
         ("Hybrid-a1", "Hybrid, $a_{max}=1$"), ("TL", "TL algorithm"), ("NSGA-III", "NSGA-III"),
         ("GNN", "GCN-EA")]


def cell(d, k, digits=3, ref=False):
    m, s, n, sym = d[k]["mean"], d[k]["std"], d[k]["n"], d[k]["sym"]
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "--"
    sym = {"+": "$^{+}$", "−": "$^{-}$", "≈": "$^{\\approx}$", "n/a": "", "": ""}[sym]
    txt = f"{m:.{digits}f} $\\pm$ {s:.{digits}f}" if k != "ps" else f"{m:.1f} $\\pm$ {s:.1f}"
    if n < 30 and k in ("igd", "sp"):
        txt += f" ($n$={n})"
    return (f"\\textbf{{{txt}}}" if ref else txt) + sym


rows = []
for key, name in ORDER:
    d = summ["algs"][key]
    ref = key == "TL-MOWOA-v2"
    rows.append(f"{name} & {d['feasible_runs']}/{d['n_runs']} & {cell(d, 'hv', ref=ref)} & {cell(d, 'igd', ref=ref)} & "
                f"{cell(d, 'sp', ref=ref)} & {cell(d, 'ps', ref=ref)} \\\\")
tab = r"""\begin{table}[t]
\centering
\caption{Final performance over 30 independent runs on the confirmatory seeds (mean $\pm$ standard deviation). Superscripts give the outcome of a two-sided Wilcoxon rank-sum test ($\alpha=0.05$) of TL-MOWOA against each algorithm: $^{+}$ TL-MOWOA significantly better, $^{-}$ significantly worse, $^{\approx}$ no significant difference. IGD and spread are computed over runs with at least one feasible solution ($n$ given where fewer than 30); runs without a feasible solution contribute HV $=0$.}
\label{tab:metrics}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\hline
\textbf{Algorithm} & \textbf{Feasible runs} & \textbf{HV $\uparrow$} & \textbf{IGD $\downarrow$} & \textbf{Spread $\downarrow$} & \textbf{Pareto size $\uparrow$} \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}%
}
\end{table}
"""
open(out_metrics, "w").write(tab)

# ------------------------------ SUMO table ------------------------------
SUMO_ORDER = [("TL-MOWOA-v2", "\\textbf{TL-MOWOA (proposed)}"), ("TL-MOWOA", "TL-MOWOA, $a_{max}=2$"),
              ("MOWOA", "MOWOA"), ("Hybrid", "Hybrid"), ("TL", "TL algorithm"), ("NSGA-III", "NSGA-III"),
              ("GNN", "GCN-EA"), ("Uniform fixed-time plan", "Uniform fixed-time plan")]
by = {}
for r in sumo:
    by.setdefault(r["alg"], []).append(r)
if "TL-MOWOA-v2" not in by:
    print("SUMO results for TL-MOWOA-v2 not available yet; metrics table only")
    sys.exit(0)
base = by["Uniform fixed-time plan"][0]
ref_tt = np.array([r["travel_time_min"] for r in by["TL-MOWOA-v2"]])
srows = []
stats = {}
for key, name in SUMO_ORDER:
    if key not in by:
        continue
    rs = by[key]
    tt = np.array([r["travel_time_min"] for r in rs]); dl = np.array([r["delay_s"] for r in rs])
    fu = np.array([r["fuel_L100km"] for r in rs]); co = np.array([r["co2_g_km"] for r in rs])
    sym = ""
    if key not in ("TL-MOWOA-v2", "Uniform fixed-time plan") and len(tt) >= 2:
        p = mannwhitneyu(ref_tt, tt, alternative="two-sided").pvalue
        sym = "$^{\\approx}$" if p >= 0.05 else ("$^{+}$" if np.median(ref_tt) < np.median(tt) else "$^{-}$")
    f = (lambda x, dg: f"{x.mean():.{dg}f} $\\pm$ {x.std(ddof=1):.{dg}f}") if len(rs) > 1 else (lambda x, dg: f"{x.mean():.{dg}f}")
    dtt = (tt.mean() / base["travel_time_min"] - 1) * 100
    dfu = (fu.mean() / base["fuel_L100km"] - 1) * 100
    stats[key] = {"n": len(rs), "tt": tt.mean(), "fuel": fu.mean(), "co2": co.mean(), "delay": dl.mean(),
                  "dtt": dtt, "dfu": dfu}
    delta = f"{dtt:+.1f}\\% / {dfu:+.1f}\\%".replace("-", "$-$")
    srows.append(f"{name} & {len(rs)} & {f(tt, 3)}{sym} & {f(dl, 1)} & {f(fu, 2)} & {f(co, 1)} & " + delta + " \\\\")
stab = r"""\begin{table}[t]
\centering
\caption{Microscopic validation in SUMO of the best compromise solution of each run (confirmatory seeds 31--40; one plan per run that found a feasible solution). Travel time includes departure delay; the last column gives the change in travel time and fuel relative to the uniform fixed-time plan (common 100~s cycle, equal splits, zero offsets). Superscripts: Wilcoxon rank-sum test of travel time against TL-MOWOA, as in Table~\ref{tab:metrics}.}
\label{tab:sumo}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\hline
\textbf{Plan} & \textbf{Plans} & \textbf{Travel time (min)} & \textbf{Delay (s/veh)} & \textbf{Fuel (L/100 km)} & \textbf{CO$_2$ (g/km)} & \textbf{$\Delta$ travel time / fuel} \\
\hline
""" + "\n".join(srows) + r"""
\hline
\end{tabular}%
}
\end{table}
"""
open(out_sumo, "w").write(stab)
json.dump(stats, open(out_sumo.replace(".tex", ".json"), "w"), indent=1)
print(tab); print(stab); print(json.dumps(stats, indent=1))
