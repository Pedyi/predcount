#!/usr/bin/env python3
"""
A LEARNED predictor for PredCount (the practical question for JEA).

So far the experiments used a "perfect" predictor (exact per-edge copy counts)
and synthetically noised versions of it. Neither is available in practice. This
module asks the operational question:

    Can a lightweight model, trained on CHEAP features computable in a stream,
    recover enough of the heavy-edge signal to deliver the speedup? And does it
    TRANSFER from one graph to another?

Features per edge {u,v} (all cheap; degrees are maintainable in one pass, core
numbers in a light preprocessing pass):
    deg(u), deg(v), min, max, product (log), sum (log),
    core(u), core(v), min-core, max-core
Target: log(1 + t(e))  where t(e) = #triangles through e   (K3 case)

Model: ridge regression on log-features (deliberately simple/fast: the point is
that even a trivial model captures most of the achievable gain; a heavier model
would only strengthen the conclusion).

Protocol:
    train on graph A -> predict on graph B (never train and test on the same
    graph), then plug the predicted weights into the exact success-probability
    machinery and report the realized speedup versus:
        - the perfect predictor (upper bound)
        - the uniform baseline (1x)

Run in Colab:
    !python learned_predictor.py
    # or: import learned_predictor as lp; lp.main()
"""
import gzip
import urllib.request
import numpy as np
import networkx as nx

SNAP = {
    "ca-GrQc":  "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
    "ca-HepTh": "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
    "facebook": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
}


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


# ---------------- features and targets ----------------
def edge_features(G):
    """Cheap, stream-friendly features per edge. Returns (edges, X)."""
    deg = dict(G.degree())
    core = nx.core_number(G)
    edges = [tuple(sorted(e, key=str)) for e in G.edges()]
    X = np.zeros((len(edges), 8))
    for i, (u, v) in enumerate(edges):
        du, dv = deg[u], deg[v]
        cu, cv = core[u], core[v]
        X[i] = [
            np.log1p(min(du, dv)),
            np.log1p(max(du, dv)),
            np.log1p(du) + np.log1p(dv),
            np.log1p(abs(du - dv)),
            np.log1p(min(cu, cv)),
            np.log1p(max(cu, cv)),
            np.log1p(cu) + np.log1p(cv),
            1.0,                       # bias
        ]
    return edges, X


def edge_targets_K3(G, edges):
    adj = {v: set(G.neighbors(v)) for v in G}
    y = np.array([np.log1p(len(adj[u] & adj[v])) for (u, v) in edges])
    return y


def fit_ridge(X, y, lam=1.0):
    A = X.T @ X + lam * np.eye(X.shape[1])
    b = X.T @ y
    return np.linalg.solve(A, b)


# ---------------- evaluation machinery (exact success prob, K3) ----------------
def success_prob_K3(G, weights=None):
    """
    weights: dict frozenset(edge)->predicted weight, or None for uniform.
    Exact per-instance success probability of the full-path sampler.
    """
    deg = dict(G.degree())
    adj = {v: set(G.neighbors(v)) for v in G}
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
                trip = sorted((u, a, b), key=lambda v: (deg[v], str(v)))
                piv, mid, close = trip
                if weights is None:
                    p += (1.0 / M) * inv
                else:
                    pb = (weights.get(frozenset((piv, mid)), 0.0) + 1.0) / W
                    wsum = 0.0
                    wc = 0.0
                    for c in adj[piv]:
                        w = weights.get(frozenset((piv, c)), 0.0) + 1.0
                        wsum += w
                        if c == close:
                            wc = w
                    p += pb * (wc / wsum if wsum > 0 else 0.0)
    return p


def evaluate(G, name, theta=None, train_stats=None):
    """Returns dict of speedups: perfect, learned (if theta given)."""
    edges, X = edge_features(G)
    y = edge_targets_K3(G, edges)
    te = {frozenset(e): np.expm1(yy) for e, yy in zip(edges, y)}
    p_u = success_prob_K3(G, None)
    p_perf = success_prob_K3(G, te)
    out = {"m": G.number_of_edges(), "perfect": p_perf / p_u}
    if theta is not None:
        pred = np.maximum(np.expm1(X @ theta), 0.0)
        wl = {frozenset(e): float(pv) for e, pv in zip(edges, pred)}
        p_learn = success_prob_K3(G, wl)
        out["learned"] = p_learn / p_u
        # quality of the prediction itself
        out["corr"] = float(np.corrcoef(X @ theta, y)[0, 1])
        out["frac_of_perfect"] = out["learned"] / out["perfect"]
    return out


def main():
    graphs = {}
    for name, url in SNAP.items():
        try:
            graphs[name] = load_snap(name, url)
        except Exception as e:
            print(f"  [skip {name}: {e}]")
    if len(graphs) < 2:
        print("need >=2 graphs for transfer evaluation")
        return

    names = list(graphs)
    print(f"\n{'train_on':>11}{'test_on':>11}{'m_test':>9}{'corr':>7}"
          f"{'sp_perfect':>12}{'sp_learned':>12}{'frac':>7}")
    for tr in names:
        edges_tr, Xtr = edge_features(graphs[tr])
        ytr = edge_targets_K3(graphs[tr], edges_tr)
        theta = fit_ridge(Xtr, ytr)
        for te_name in names:
            if te_name == tr:
                continue                     # strict transfer: never test on train
            r = evaluate(graphs[te_name], te_name, theta)
            print(f"{tr:>11}{te_name:>11}{r['m']:>9}{r['corr']:>7.2f}"
                  f"{r['perfect']:>12.1f}{r['learned']:>12.1f}"
                  f"{r['frac_of_perfect']:>7.2f}")

    print("\nReading:")
    print("  sp_perfect  = speedup with exact per-edge triangle counts (upper bound)")
    print("  sp_learned  = speedup with the ridge model trained on ANOTHER graph")
    print("  frac        = learned / perfect; how much of the achievable gain a")
    print("                cheap, transferable model recovers.")
    print("  corr        = correlation of predicted vs true log-heaviness on the")
    print("                test graph (prediction quality, i.e. small distortion).")


if __name__ == "__main__":
    main()
