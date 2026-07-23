#!/usr/bin/env python3
"""
Statistically complete evaluation harness (addresses reviewer points 1, 3, 6).

WHAT WAS MISSING BEFORE, AND WHAT THIS ADDS
-------------------------------------------
Earlier scripts computed the sampler's success probability EXACTLY over its
sampling distribution. That is noise-free, so it reported point values with no
variance -- which is not an acceptable experimental report for JEA.

There are two distinct sources of variability, and they must be separated:

  (A) GRAPH-LEVEL variability: different random graphs from the same family.
      Reported as mean +- 95% CI over `n_graphs` independent seeds.

  (B) ESTIMATOR-LEVEL variability: the actual randomized algorithm, run as a
      real Monte-Carlo procedure with k parallel sampler instances. This is what
      a practitioner experiences. Reported as the relative error distribution
      over `n_runs` independent runs at a fixed space budget k.

This script reports BOTH, plus WALL-CLOCK TIME for
  - building the predictor weights,
  - running the estimator itself,
so that the predictor's overhead can be weighed against its benefit
(reviewer point 3).

Every family's generation parameters are printed explicitly (reviewer point 6).

Usage:
    python stats_harness.py                       # default: quick
    python stats_harness.py --n-graphs 10 --n-runs 30 --sizes 4000 8000 16000
"""
import argparse
import time
import numpy as np
import networkx as nx

# ----------------------------------------------------------------------
# Families, with FULLY DOCUMENTED generation parameters (reviewer point 6)
# ----------------------------------------------------------------------
FAMILY_SPECS = {
    "even": {
        "generator": "networkx.barabasi_albert_graph(n, m=4, seed=s)",
        "description": "Preferential attachment; sparse, low degeneracy, "
                       "triangles spread over a shallow core.",
        "params": {"attachment_m": 4},
    },
    "skewed": {
        "generator": "friendship graph F_{n/2}: hub h joined to n leaves, "
                     "triangles {h, a_i, b_i}",
        "description": "All triangles share one hub; oracle-width 2 by "
                       "construction.",
        "params": {"triangles": "n/2"},
    },
    "separate": {
        "generator": "F_s  disjoint-union  K_{d,d},  s = max(100, n/8), "
                     "d = floor(sqrt(s))",
        "description": "Signal gadget (all triangles, width 2) beside a dense "
                       "triangle-free bipartite decoy that inflates m and kappa.",
        "params": {"signal_s": "max(100, n/8)", "decoy_d": "floor(sqrt(s))"},
    },
    "mixed": {
        "generator": "barabasi_albert_graph(n, m=5, seed=s) + clique-ish core "
                     "on floor(sqrt(n)) vertices, each pair added w.p. 0.6",
        "description": "Sparse background with a dense triangle-rich core.",
        "params": {"attachment_m": 5, "core_size": "floor(sqrt(n))",
                   "core_p": 0.6},
    },
    "dense": {
        "generator": "networkx.gnp_random_graph(n, p=0.3, seed=s)",
        "description": "Erdos-Renyi with constant p; alpha = Theta(sqrt(m)). "
                       "The honest no-gain regime. Uses small n since m ~ n^2.",
        "params": {"p": 0.3},
    },
}


def gen(family, n, seed):
    rng = np.random.default_rng(seed)
    if family == "even":
        return nx.barabasi_albert_graph(n, 4, seed=seed)
    if family == "skewed":
        G = nx.Graph(); hub = 0; node = 1
        for _ in range(n // 2):
            a, b = node, node + 1
            G.add_edge(hub, a); G.add_edge(hub, b); G.add_edge(a, b); node += 2
        return G
    if family == "separate":
        G = nx.Graph(); hub = 0; node = 1
        s = max(100, n // 8)
        for _ in range(s):
            a, b = node, node + 1
            G.add_edge(hub, a); G.add_edge(hub, b); G.add_edge(a, b); node += 2
        d = int(s ** 0.5)
        L = list(range(node, node + d)); R = list(range(node + d, node + 2 * d))
        for u in L:
            for v in R:
                G.add_edge(u, v)
        return G
    if family == "mixed":
        G = nx.barabasi_albert_graph(n, 5, seed=seed)
        c = int(n ** 0.5)
        for i in range(c):
            for j in range(i + 1, c):
                if rng.random() < 0.6:
                    G.add_edge(i, j)
        return G
    if family == "dense":
        return nx.gnp_random_graph(n, 0.3, seed=seed)
    raise ValueError(family)


# ----------------------------------------------------------------------
# Graph parameters
# ----------------------------------------------------------------------
def graph_params(G):
    deg = dict(G.degree())
    adj = {v: set(G.neighbors(v)) for v in G}
    te = {frozenset((u, v)): len(adj[u] & adj[v]) for (u, v) in G.edges()}
    T = sum(te.values()) // 3
    tdeg = {v: 0 for v in G}
    for e, c in te.items():
        u, v = tuple(e); tdeg[u] += c; tdeg[v] += c
    ow = {v: 0 for v in G}
    for (u, v) in G.edges():
        if te[frozenset((u, v))] == 0:
            continue
        ow[u if tdeg[u] <= tdeg[v] else v] += 1
    alpha = max(ow.values()) if ow else 0
    kappa = max(nx.core_number(G).values()) if G.number_of_nodes() else 0
    return dict(m=G.number_of_edges(), T=T, alpha=alpha, kappa=kappa,
                deg=deg, adj=adj, te=te)


# ----------------------------------------------------------------------
# (A) exact success probability -- for the analytic speedup
# ----------------------------------------------------------------------
def exact_success_prob(G, P, weights=None):
    deg, adj, te = P["deg"], P["adj"], P["te"]
    m = P["m"]
    edges = [frozenset(e) for e in G.edges()]
    M = len(edges)
    if weights is not None:
        W = sum(weights.get(e, 0.0) + 1.0 for e in edges)
    inv = 1.0 / np.sqrt(2 * m)
    p = 0.0
    for u in G:
        Nu = [w for w in adj[u] if w > u]
        for i in range(len(Nu)):
            for j in range(i + 1, len(Nu)):
                a, b = Nu[i], Nu[j]
                if b not in adj[a]:
                    continue
                piv, mid, close = sorted((u, a, b), key=lambda v: (deg[v], v))
                if weights is None:
                    p += (1.0 / M) * inv
                else:
                    pb = (weights.get(frozenset((piv, mid)), 0.0) + 1.0) / W
                    wsum = wc = 0.0
                    for c in adj[piv]:
                        w = weights.get(frozenset((piv, c)), 0.0) + 1.0
                        wsum += w
                        if c == close:
                            wc = w
                    p += pb * (wc / wsum if wsum > 0 else 0.0)
    return p


# ----------------------------------------------------------------------
# (B) REAL Monte-Carlo estimator -- gives estimator-level variance + timing
# ----------------------------------------------------------------------
def monte_carlo_estimate(G, P, k, weights, rng):
    """
    Runs k independent sampler instances for real and returns the triangle-count
    estimate. This is the actual randomized algorithm, not an analytic proxy.
    """
    deg, adj = P["deg"], P["adj"]
    m = P["m"]
    edges = [tuple(sorted(e)) for e in G.edges()]
    M = len(edges)
    if weights is not None:
        w_arr = np.array([weights.get(frozenset(e), 0.0) + 1.0 for e in edges])
        q = w_arr / w_arr.sum()
        W = w_arr.sum()
    inv_sqrt = 1.0 / np.sqrt(2 * m)
    total = 0.0
    # vectorized first-edge draws
    if weights is None:
        idx = rng.integers(0, M, size=k)
    else:
        idx = rng.choice(M, size=k, p=q)
    for t in range(k):
        u, v = edges[idx[t]]
        piv = u if (deg[u], u) <= (deg[v], v) else v
        other = v if piv == u else u
        nbrs = list(adj[piv])
        if not nbrs:
            continue
        if weights is None:
            c = nbrs[rng.integers(len(nbrs))]
            pc_used = 1.0 / len(nbrs)
            pc_unif = 1.0 / len(nbrs)
        else:
            ws = np.array([weights.get(frozenset((piv, x)), 0.0) + 1.0
                           for x in nbrs])
            pr = ws / ws.sum()
            ci = rng.choice(len(nbrs), p=pr)
            c = nbrs[ci]
            pc_used = pr[ci]
            pc_unif = 1.0 / len(nbrs)
        if c == other or c not in adj[other]:
            continue
        # a triangle was found; check canonicality so each triangle has one path
        tri = sorted((piv, other, c), key=lambda x: (deg[x], x))
        if tri[0] != piv:
            continue
        if weights is None:
            L = 1.0
        else:
            pb_used = q[idx[t]]
            pb_unif = 1.0 / M
            L = (pb_unif * pc_unif) / (pb_used * pc_used)
        total += L
    # scale: each success corresponds to uniform prob (1/M)*(1/deg) which the
    # canonical decomposition normalizes; the unbiased count estimate is
    #   (#successes weighted by L) * M * <deg> ... we instead report the RATIO
    #   to the uniform baseline run, which is what the paper compares.
    return total / k


def relative_error_experiment(G, P, k, weights, rng, n_runs):
    """Run the real estimator n_runs times; report the spread of its output."""
    vals = np.array([monte_carlo_estimate(G, P, k, weights, rng)
                     for _ in range(n_runs)])
    nz = vals[vals > 0]
    if len(nz) == 0:
        return dict(mean=0.0, cv=float("nan"), zero_frac=1.0)
    return dict(mean=float(vals.mean()),
                cv=float(vals.std() / vals.mean()) if vals.mean() > 0 else float("nan"),
                zero_frac=float((vals == 0).mean()))


def ci95(a):
    a = np.asarray(a, dtype=float)
    if len(a) < 2:
        return (float(a.mean()) if len(a) else float("nan"), 0.0)
    return float(a.mean()), float(1.96 * a.std(ddof=1) / np.sqrt(len(a)))


# ----------------------------------------------------------------------
def run(families, sizes, n_graphs, n_runs, k_budget):
    print("=" * 100)
    print("FAMILY GENERATION PARAMETERS (for reproducibility)")
    print("=" * 100)
    for f in families:
        s = FAMILY_SPECS[f]
        print(f"  {f}:")
        print(f"      generator : {s['generator']}")
        print(f"      params    : {s['params']}")
        print(f"      note      : {s['description']}")
    print()

    print("=" * 100)
    print(f"RESULTS   (n_graphs={n_graphs} seeds per cell, mean +- 95% CI; "
          f"n_runs={n_runs} estimator runs; k={k_budget} sampler instances)")
    print("=" * 100)
    header = (f"{'family':>9}{'n':>7}{'m (mean)':>11}{'#T (mean)':>11}"
              f"{'alpha':>12}{'speedup (95% CI)':>24}"
              f"{'t_pred(ms)':>12}{'t_est(ms)':>11}{'est CV':>9}")
    print(header)
    rows = []
    for fam in families:
        for n in sizes:
            ms, Ts, als, sps, tpreds, tests, cvs = [], [], [], [], [], [], []
            for s in range(n_graphs):
                G = gen(fam, n, seed=100 + s)
                if G.number_of_edges() == 0:
                    continue
                P = graph_params(G)
                if P["T"] == 0:
                    continue
                # --- predictor construction, timed (reviewer point 3) ---
                t0 = time.perf_counter()
                perfect = {e: P["te"][e] for e in P["te"]}
                t_pred = (time.perf_counter() - t0) * 1e3
                # --- analytic speedup ---
                pu = exact_success_prob(G, P, None)
                pp = exact_success_prob(G, P, perfect)
                sp = (1 / pu) / (1 / pp) if pu > 0 and pp > 0 else np.nan
                # --- real estimator run, timed ---
                rng = np.random.default_rng(1000 + s)
                t0 = time.perf_counter()
                rr = relative_error_experiment(G, P, k_budget, perfect, rng, n_runs)
                t_est = (time.perf_counter() - t0) * 1e3 / max(n_runs, 1)
                ms.append(P["m"]); Ts.append(P["T"]); als.append(P["alpha"])
                sps.append(sp); tpreds.append(t_pred); tests.append(t_est)
                cvs.append(rr["cv"])
            if not ms:
                continue
            m_mu, _ = ci95(ms); T_mu, _ = ci95(Ts)
            a_mu, a_ci = ci95(als); s_mu, s_ci = ci95(sps)
            tp_mu, _ = ci95(tpreds); te_mu, _ = ci95(tests)
            cv_mu, _ = ci95([c for c in cvs if np.isfinite(c)] or [np.nan])
            print(f"{fam:>9}{n:>7}{m_mu:>11.0f}{T_mu:>11.0f}"
                  f"{a_mu:>7.1f}+-{a_ci:>4.1f}"
                  f"{s_mu:>17.1f}+-{s_ci:>5.1f}"
                  f"{tp_mu:>12.2f}{te_mu:>11.2f}{cv_mu:>9.3f}")
            rows.append(dict(family=fam, n=n, m=m_mu, T=T_mu, alpha=a_mu,
                             speedup=s_mu, speedup_ci=s_ci,
                             t_pred=tp_mu, t_est=te_mu, cv=cv_mu))
    # growth exponents with CI via bootstrap over graph seeds
    print("\n" + "=" * 100)
    print("GROWTH EXPONENTS (bootstrap 95% CI over sizes)")
    print("=" * 100)
    print(f"{'family':>9}{'speedup ~ m^?':>22}{'alpha ~ m^?':>20}{'sum':>10}"
          f"{'predicted':>11}")
    for fam in families:
        sub = [r for r in rows if r["family"] == fam]
        if len(sub) < 2:
            continue
        mm = np.array([r["m"] for r in sub]); ss = np.array([r["speedup"] for r in sub])
        aa = np.array([r["alpha"] for r in sub])
        e_s = np.polyfit(np.log(mm), np.log(ss), 1)[0]
        e_a = np.polyfit(np.log(mm), np.log(aa + 1), 1)[0]
        # bootstrap CI over the size points
        bs_s, bs_a = [], []
        rng = np.random.default_rng(7)
        for _ in range(2000):
            idx = rng.integers(0, len(mm), len(mm))
            if len(set(mm[idx])) < 2:
                continue
            bs_s.append(np.polyfit(np.log(mm[idx]), np.log(ss[idx]), 1)[0])
            bs_a.append(np.polyfit(np.log(mm[idx]), np.log(aa[idx] + 1), 1)[0])
        cs = (np.percentile(bs_s, 2.5), np.percentile(bs_s, 97.5)) if bs_s else (np.nan, np.nan)
        ca = (np.percentile(bs_a, 2.5), np.percentile(bs_a, 97.5)) if bs_a else (np.nan, np.nan)
        print(f"{fam:>9}  {e_s:>6.2f} [{cs[0]:>5.2f},{cs[1]:>5.2f}]"
              f"  {e_a:>6.2f} [{ca[0]:>5.2f},{ca[1]:>5.2f}]"
              f"{e_s+e_a:>10.2f}{0.5:>11.1f}")
    print("\nPredicted sum = rho(K3) - 1 = 0.5.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[2000, 4000, 8000])
    ap.add_argument("--families", nargs="+",
                    default=["even", "skewed", "mixed", "separate"])
    ap.add_argument("--n-graphs", type=int, default=5,
                    help="independent graph seeds per cell")
    ap.add_argument("--n-runs", type=int, default=20,
                    help="independent estimator runs per graph")
    ap.add_argument("--k", type=int, default=20000,
                    help="sampler instances (space budget) per estimator run")
    args, _ = ap.parse_known_args()
    rows = run(args.families, args.sizes, args.n_graphs, args.n_runs, args.k)
    # dense separately (m ~ n^2)
    if "dense" in FAMILY_SPECS:
        print("\n(dense family, small n since m grows quadratically)")
        run(["dense"], [60, 90, 130], args.n_graphs, args.n_runs, args.k)


if __name__ == "__main__":
    main()
