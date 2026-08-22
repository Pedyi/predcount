#!/usr/bin/env bash
# Runs every numerical check that backs a claim in the paper.
# Offline-safe: real-graph scripts are skipped if there is no network.
set -u
cd "$(dirname "$0")"

echo "##### THEORY CHECKS #####"
echo "=== Prop 1: unbiasedness for all predictor types ==="
python3 experiments/proof_foundation.py
echo; echo "=== Thm 4: separation, O(1) vs Omega(sqrt m) ==="
python3 experiments/verify_separation.py
echo; echo "=== Lem A: worst-case pivot weight delta* <= tau^2 (NOT Theta(tau)) ==="
python3 experiments/verify_pivot.py
echo; echo "=== Lem A: averaged pivot weight delta_bar = Theta(alpha) ==="
python3 experiments/verify_pivot_avg.py
echo; echo "=== App A: separability parameter tau ==="
python3 experiments/verify_appendixA.py
echo; echo "=== App B: exponent gap rho(H)-1 for C4, K4, C5 ==="
python3 experiments/verify_generalH.py
echo; echo "=== App B: per-part exponent accounting (exact rho-1) ==="
python3 experiments/verify_appendixB.py

echo; echo "##### EXPERIMENTS (with statistics) #####"
echo "=== Table 1: synthetic map with CIs, timing, estimator variance ==="
python3 experiments/stats_harness.py --sizes 2000 4000 --n-graphs 5 --n-runs 20
echo; echo "=== Table 3: comparison vs reimplemented competitors ==="
python3 experiments/compare_methods.py --sizes 2000 4000
echo; echo "=== C4 synthetic sweep ==="
python3 src/c4_ext.py --sizes 400 800 1600
echo; echo "=== C4 proxy sensitivity (exact vs codegree) ==="
python3 experiments/sensitivity_c4.py

echo; echo "##### LEARNED PREDICTOR #####"
echo "=== leave-one-graph-out CV, permutation test, ablation ==="
python3 experiments/validate_learned.py
echo; echo "=== structural vs distributional decomposition ==="
python3 experiments/permutation_control.py

echo; echo "##### REAL GRAPHS (needs internet) #####"
python3 experiments/real_graphs.py   || echo "  [skipped: no network]"
python3 experiments/c4_real.py       || echo "  [skipped: no network]"
python3 experiments/extended_real.py --skip-large || echo "  [skipped: no network]"
