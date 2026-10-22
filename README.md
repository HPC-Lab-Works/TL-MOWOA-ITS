# TL-MOWOA: reproducible experiment code

Code for the experiments in *Multi-Objective Control Parameter Optimization in
Intelligent Transportation Systems using Whale Optimization Algorithm and
enhanced with Transfer Learning*.

## Problem
`problem.py` — signal-timing optimization on a 4 × 4 grid of signalized
intersections (400 m two-lane links, 50 km/h).

* 48 decision variables: per intersection, cycle length C ∈ [40, 180] s, N–S
  green split ∈ [0.15, 0.85] and offset fraction ∈ [0, 1).
* Objectives (minimized): average travel time (min), congestion index
  (flow-weighted mean degree of saturation), CO₂ (g/km), fuel (L/100 km).
* Constraints: lane back-of-queue ≤ 30 veh; 60 ≤ C ≤ 150 s and effective
  green ≥ 10 s; degree of saturation ≤ 1 on every approach.
* Evaluation: HCM-style uniform delay with an offset-based progression factor
  plus incremental delay; Akçelik elemental fuel model; CO₂ from fuel with an
  85 % petrol / 15 % diesel fleet (CO₂ is therefore proportional to fuel).
* Target task: heavy morning peak. Source tasks (historical): off-peak,
  shoulder, evening peak (reversed direction) and the previous year's
  morning peak.

## Algorithms (`algorithms.py`, `tl.py`)
All algorithms receive the same budget of 5,000 true target evaluations and
keep the same external archive (M = 100).

| Name | Description |
|---|---|
| TL-MOWOA | TL initialization (p_TL), TL-guided leaders (p_TL), adaptive controller (τ_C, τ_D) |
| MOWOA | Multi-objective WOA with crowding-distance leader roulette and constraint dominance |
| Hybrid | TL initialization followed by standard MOWOA (ablation) |
| TL | Transfer-only search: sampling from priors, latent regions and archive perturbations |
| NSGA-III | pymoo NSGA-III, Das–Dennis reference directions (p = 5) |
| GCN-EA | NSGA-II-style EA with a graph-convolutional surrogate pre-screening 5N offspring |

Transfer learning: a source MLP surrogate is pre-trained on all 20,000
evaluations from the four source optimizations, then fine-tuned on 200 Latin
hypercube target samples (counted in the budget) with
MSE + β·MMD² + γ·mean‖θ − θ_s‖² (β = 0.57, γ = 0.39). Priors (k = 7) are the
source archive solutions with the best predicted constrained non-domination
rank on the target, with ties broken by kernel-mean-matching weight. Three
latent regions come from k-means on the embeddings.

Parameters of TL-MOWOA (p_TL = 0.1, τ_C = 1.0, τ_D = 0.2) were selected on
tuning seeds 101–103 (`tune.py`), which are disjoint from evaluation seeds 1–30.

## Reproduce
```bash
pip install numpy scipy scikit-learn torch pymoo matplotlib eclipse-sumo traci sumolib
cd code && python build_source.py out/source.pkl
python run.py out/source.pkl out/runs MOWOA,Hybrid,TL-MOWOA,TL,NSGA-III,GNN 1-30 \
  '{"p_tl": 0.1, "tau_c": 1.0, "tau_d": 0.2}'
python analyze.py out/runs out/analysis
python ../sumo_val/validate.py out/runs out/analysis/curves.pkl out/sumo.json 1-10
```

## Metrics
The reference front P* is the non-dominated union of all final feasible fronts.
Objectives are normalized by the ideal and nadir points of P*. Metrics are HV
(reference point 1.1), IGD to P*, generalized spread (Zhou et al., 2006) and
the number of feasible non-dominated solutions. Each baseline is compared with
TL-MOWOA using a two-sided Wilcoxon rank-sum test (α = 0.05).

## Rounds and naming
* `TL-MOWOA-v2` in the code is the method reported as **TL-MOWOA** in the
  paper (coefficient schedule with a_max = 1). `TL-MOWOA` in the code is the
  original configuration (a_max = 2), reported as "TL-MOWOA, a_max = 2".
* Round 1 (seeds 1–30): original configuration against the five baselines.
* Development (seeds 101–108): `diag.py` (move-type diagnosis) and `tune2.py`
  (eight variants); all results are in `results/tuning_round2.json`.
* Round 2, confirmatory (seeds 31–60, never used in development): all
  algorithms plus the ablations `MOWOA-a1` and `Hybrid-a1`. The results tables are
  generated from `results/round2_confirmatory_seeds31-60/` by `gen_tables.py`.
* `raw_runs/` holds zipped per-run archives and per-generation fronts (pickle):
  `raw_runs_round1.zip`, `raw_runs_round2.zip`, and `source_tasks.zip` (the
  historical source optimization data D_s). Unzip before running `analyze.py`.
