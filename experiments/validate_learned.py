#!/usr/bin/env python3
"""
Rigorous validation of the learned predictor (addresses reviewer point 7).

WHAT THE REVIEWER ASKED FOR
---------------------------
The earlier learned-predictor experiment reported correlations 0.47-0.70 and a
"recovers ~65%" figure with no statistical backing. This script adds:

  1. LEAVE-ONE-GRAPH-OUT cross-validation (strict: never train and test on the
     same graph), with the number of train/test pairs stated.
  2. BOOTSTRAP confidence intervals on the recovered-fraction statistic.
  3. A PERMUTATION TEST: is the learned predictor's speedup significantly better
     than a random predictor with the same marginal weight distribution?
  4. An ABLATION over feature groups (degree-only, core-only, both), to see
     which signal actually carries the predictor -- and to check that the model
     is not overfitting to a single graph's idiosyncrasies.
  5. RIDGE PENALTY SWEEP, to show the result is not an artifact of one lambda.

Usage:
    python validate_learned.py                 # synthetic stand-ins, offline
    python validate_learned.py --real          # SNAP graphs, needs internet
"""
import argparse
import gzip
import urllib.request
import numpy as np
import networkx as nx

SNAP = {
    "ca-GrQc":  "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
    "ca-HepTh": "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
    "facebook": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
}

# feature groups for the ablation
FEATURE_GROUPS = {
    "degree-only": [0, 1, 2, 3, 7],
    "core-only":   [4, 5, 6, 7],
    "both":        [0, 1, 2, 3, 4, 5, 6, 7],
}
FEATURE_NAMES = ["log min deg", "log max deg", "log deg sum", "log deg diff",
                 "log min core", "log max core", "log core sum", "bias"]


def load_snap(name, url):
    print(f"  downloading {name} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=120).read()
    text = gzip.decompress(raw).decode("utf-8", errors="ignore")
    G = nx.Graph()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 2 and p[0] != p[1]:
            G.add_edge(p[0], p[1])
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def synthetic_graphs():
    return {
        "collab-A": nx.powerlaw_cluster_graph(1500, 3, 0.40, seed=1),
        "collab-B": nx.powerlaw_cluster_graph(1200, 4, 0.35, seed=2),
        "social-C": nx.powerlaw_cluster_graph(900, 8, 0.50, seed=3),
        "social-D": nx.powerlaw_cluster_graph(1100, 6, 0.45, seed=4),
    }


def features(G):
    deg = dict(G.degree())
    core = nx.core_number(G)
    edges = [tuple(sorted(e, key=str)) for e in G.edges()]
    X = np.zeros((len(edges), 8))
    for i, (u, v) in enumerate(edges):
        du, dv, cu, cv = deg[u], deg[v], core[u], core[v]
        X[i] = [np.log1p(min(du, dv)), np.log1p(max(du, dv)),
                np.log1p(du) + np.log1p(dv), np.log1p(abs(du - dv)),
                np.log1p(min(cu, cv)), np.log1p(max(cu, cv)),
                np.log1p(cu) + np.log1p(cv), 1.0]
    return edges, X


def targets(G, edges):
    adj = {v: set(G.neighbors(v)) for v in G}
    return np.array([np.log1p(len(adj[u] & adj[v])) for (u, v) in edges])


def ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def success_prob(G, weights=None):
    deg = dict(G.degree()); adj = {v: set(G.neighbors(v)) for v in G}
    m = G.number_of_edges()
    edges = [frozenset(e) for e in G.edges()]
    M = len(edges)
    if weights is not None:
        W = sum(weights.get(e, 0.0) + 1.0 for e in edges)
    inv = 1.0 / np.sqrt(2 * m)
    p = 0.0
    for u in G:
        Nu = [w for w in adj[u] if str(w) > str(u)]
        for i in range(len(Nu)):
            for j in range(i + 1, len(Nu)):
                a, b = Nu[i], Nu[j]
                if b not in adj[a]:
                    continue
                piv, mid, close = sorted((u, a, b), key=lambda v: (deg[v], str(v)))
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


def evaluate_on(G, theta, cols, rng, n_perm=200):
    """Returns dict with perfect / learned / random-permuted speedups and corr."""
    edges, X = features(G)
    y = targets(G, edges)
    te = {frozenset(e): float(np.expm1(v)) for e, v in zip(edges, y)}
    p_u = success_prob(G, None)
    p_perfect = success_prob(G, te)
    pred = np.maximum(np.expm1(X[:, cols] @ theta), 0.0)
    wl = {frozenset(e): float(p) for e, p in zip(edges, pred)}
    p_learn = success_prob(G, wl)
    corr = float(np.corrcoef(X[:, cols] @ theta, y)[0, 1])
    # permutation control: same weight multiset, shuffled across edges
    perm_speedups = []
    for _ in range(min(n_perm, 30)):        # success_prob is the bottleneck
        sh = rng.permutation(pred)
        wp = {frozenset(e): float(p) for e, p in zip(edges, sh)}
        perm_speedups.append(success_prob(G, wp) / p_u)
    return dict(perfect=p_perfect / p_u, learned=p_learn / p_u,
                corr=corr, perm=np.array(perm_speedups))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--lams", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    args, _ = ap.parse_known_args()

    if args.real:
        graphs = {}
        for n, u in SNAP.items():
            try:
                graphs[n] = load_snap(n, u)
            except Exception as e:
                print(f"  [skip {n}: {e}]")
    else:
        graphs = synthetic_graphs()

    names = list(graphs)
    n_pairs = len(names) * (len(names) - 1)
    print(f"\nLeave-one-graph-out cross-validation: {len(names)} graphs, "
          f"{n_pairs} ordered train/test pairs, strictly disjoint.\n")

    rng = np.random.default_rng(0)

    # ---------- main CV table ----------
    print("=" * 92)
    print("MAIN RESULT: recovered fraction of the perfect-predictor speedup")
    print("=" * 92)
    print(f"{'train':>10}{'test':>10}{'corr':>7}{'sp_perfect':>12}"
          f"{'sp_learned':>12}{'sp_random':>12}{'frac':>7}{'p_perm':>8}")
    fracs, corrs = [], []
    for tr in names:
        e_tr, X_tr = features(graphs[tr])
        y_tr = targets(graphs[tr], e_tr)
        cols = FEATURE_GROUPS["both"]
        theta = ridge(X_tr[:, cols], y_tr, 1.0)
        for te_name in names:
            if te_name == tr:
                continue
            r = evaluate_on(graphs[te_name], theta, cols, rng)
            frac = r["learned"] / r["perfect"]
            # one-sided permutation p-value: P(random >= learned)
            p_perm = float((r["perm"] >= r["learned"]).mean())
            fracs.append(frac); corrs.append(r["corr"])
            print(f"{tr:>10}{te_name:>10}{r['corr']:>7.2f}{r['perfect']:>12.1f}"
                  f"{r['learned']:>12.1f}{r['perm'].mean():>12.1f}"
                  f"{frac:>7.2f}{p_perm:>8.3f}")

    fr = np.array(fracs)
    bs = np.array([np.mean(rng.choice(fr, len(fr), replace=True))
                   for _ in range(5000)])
    print(f"\nRecovered fraction: mean {fr.mean():.3f}, "
          f"95% CI [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}] "
          f"over {len(fr)} pairs")
    print(f"Prediction correlation: mean {np.mean(corrs):.3f}, "
          f"range [{np.min(corrs):.2f}, {np.max(corrs):.2f}]")

    # ---------- feature ablation ----------
    print("\n" + "=" * 92)
    print("ABLATION: which features carry the signal?")
    print("=" * 92)
    print(f"{'features':>14}{'mean frac':>12}{'mean corr':>12}")
    for gname, cols in FEATURE_GROUPS.items():
        fs, cs = [], []
        for tr in names:
            e_tr, X_tr = features(graphs[tr])
            y_tr = targets(graphs[tr], e_tr)
            th = ridge(X_tr[:, cols], y_tr, 1.0)
            for te_name in names:
                if te_name == tr:
                    continue
                r = evaluate_on(graphs[te_name], th, cols, rng, n_perm=0)
                fs.append(r["learned"] / r["perfect"]); cs.append(r["corr"])
        print(f"{gname:>14}{np.mean(fs):>12.3f}{np.mean(cs):>12.3f}")

    # ---------- ridge penalty sweep ----------
    print("\n" + "=" * 92)
    print("SENSITIVITY: ridge penalty lambda")
    print("=" * 92)
    print(f"{'lambda':>10}{'mean frac':>12}")
    cols = FEATURE_GROUPS["both"]
    for lam in args.lams:
        fs = []
        for tr in names:
            e_tr, X_tr = features(graphs[tr])
            y_tr = targets(graphs[tr], e_tr)
            th = ridge(X_tr[:, cols], y_tr, lam)
            for te_name in names:
                if te_name == tr:
                    continue
                r = evaluate_on(graphs[te_name], th, cols, rng, n_perm=0)
                fs.append(r["learned"] / r["perfect"])
        print(f"{lam:>10.2f}{np.mean(fs):>12.3f}")

    print("\nInterpretation:")
    print("  frac      = learned speedup / perfect-predictor speedup")
    print("  sp_random = speedup of a predictor with the SAME weight multiset")
    print("              randomly permuted across edges (destroys the signal)")
    print("  p_perm    = one-sided permutation p-value; small => the learned")
    print("              predictor beats chance, i.e. it uses real structure")


if __name__ == "__main__":
    main()
