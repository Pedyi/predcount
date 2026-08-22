# PredCount — Predictions in Streaming Subgraph Counting

Reference implementation and reproducibility scripts for the paper
*"When Do Predictions Help Streaming Subgraph Counting? A Prediction-Augmented
Algorithm and an Empirical Map."*

The code computes, exactly and without Monte-Carlo noise, the per-instance
success probability of the prediction-augmented (full-path importance-weighted)
sampler and the prediction-free Fichtenberger–Peng baseline, and from these the
speedup, the graph parameters, and every numerical claim in the paper.

## Requirements

```
python >= 3.8
numpy >= 1.21
networkx >= 2.8
```

Install with `pip install -r requirements.txt`.

## Layout

```
src/
  alpha_exact.py        exact oracle-width alpha_H (Definition 3) via max-flow
  predcount.py          K3 full-path estimator + synthetic map
  c4_ext.py             C4 extension
  learned_predictor.py  ridge model over degree/core features (leave-one-graph-out)
experiments/
  proof_foundation.py   Prop. (unbiasedness) for perfect/random/adversarial predictors
  verify_separation.py  Thm. (separation): O(1) vs Omega(sqrt m)
  verify_pivot.py       Lem. (pivot-weight bound delta* = Theta(tau))
  verify_appendixA.py   separability parameter tau
  verify_generalH.py    Appendix B: exponent gap rho(H)-1 for C4, K4, C5
  verify_appendixB.py   Appendix B: per-part exponent accounting
  stats_harness.py      Table 1: synthetic map with bootstrap CIs, timing, variance
  compare_methods.py    Table 3: vs reimplemented storage/heaviness baselines
  sensitivity_c4.py     C4 codegree-proxy vs exact sensitivity
  permutation_control.py  structural vs distributional decomposition
  validate_learned.py   leave-one-graph-out CV, permutation test, ablation
  real_graphs.py        SNAP real graphs (K3, C4)  [needs internet]
  c4_real.py            SNAP real graphs (C4)       [needs internet]
  extended_real.py      the eleven-dataset table (K3, C4, sampled K4) [needs internet]
run_all_checks.sh       runs every check that backs a claim (offline-safe)
```

## Reproducing the paper

```
bash run_all_checks.sh
```

Real-graph scripts download SNAP edge lists on first run; if there is no
network they are skipped. To reproduce the eleven-dataset table (Table 2) once
network is available:

```
python3 experiments/extended_real.py
```

## Note on the oracle-width computation

`alpha_H` (Definition 3) is the least, over orientations of the copy-bearing
edges, of the maximum out-degree — equivalently the pseudoarboricity
`ceil(max_S |E(S)|/|S|)` of the copy-bearing subgraph. It is computed **exactly**
in `src/alpha_exact.py` by an integer binary search over max-flow feasibility
(the standard maximum-density-subgraph reduction), which is guaranteed to return
a value `<= kappa` (degeneracy).

Every script that needs the oracle-width imports `alpha_from_cb_edges` from this
module. An earlier version estimated `alpha_H` by the maximum out-degree of a
single greedy orientation; that is only an upper bound and can exceed `kappa`,
which is impossible for the true parameter. Routing all computations through the
exact routine fixes the reported numbers and keeps them consistent with the
theory (`alpha_H/kappa <= 1` on every graph). Each call asserts `alpha_H <= kappa`.

## What the estimator does

For each fixed copy the sampler follows one canonical path (base edges and, for
odd cycles, a closing vertex). PredCount weights every elementary choice by the
predictor and corrects the output by the likelihood ratio of the taken path, with
a 1/2 mixture against the unweighted sampler for unconditional robustness. The
estimator is unbiased for every predictor; consistent (space exponent in `m`
drops from `rho(H)` to `rho(H)-1` on graphs where copies sit on a shallow
substructure, with the alpha-exponent equal to the number of base-edge draws
`b_H`); and robust (never more than twice the prediction-free space).
