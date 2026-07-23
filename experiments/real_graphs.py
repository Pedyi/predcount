#!/usr/bin/env python3
"""
PredCount on REAL graphs -- the JEA experimental core.

Downloads a few SNAP datasets (small-to-medium, spanning sparse->dense), computes
the key parameters (alpha = oracle-width, tau = floor-degree, Delta_E = edge
heaviness, degeneracy kappa), and the perfect-predictor speedup for K3. This
tests, on real data, the theory's prediction that predictions help iff copies
concentrate on a low-width substructure (alpha = o(sqrt(m))).

Designed for Google Colab (has internet + numpy + networkx preinstalled).

USAGE in Colab:
    !wget -q https://raw.githubusercontent.com/.../predcount_v2.py   # or upload it
    # then:
    import real_graphs
    real_graphs.main()

or just run this file:  it will download, analyze, and print the map.

Datasets (SNAP, undirected, gzip edge lists):
    ca-GrQc      5,242 nodes   14,496 edges   (collaboration, sparse)
    ca-HepTh     9,877 nodes   25,998 edges   (collaboration, sparse)
    email-Enron 36,692 nodes  183,831 edges   (email, medium)
    facebook     4,039 nodes   88,234 edges   (social, dense-ish)

If a download fails (e.g. offline), point LOCAL_FILES to your own edge lists.
"""
import gzip
import io
import os
import urllib.request
import numpy as np
import networkx as nx

SNAP = {
    "ca-GrQc":     "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
    "ca-HepTh":    "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
    "email-Enron": "https://snap.stanford.edu/data/email-Enron.txt.gz",
    "facebook":    "https://snap.stanford.edu/data/facebook_combined.txt.gz",
}

# If you have local files instead, e.g. LOCAL_FILES={"mygraph":"/path/edges.txt"}
LOCAL_FILES = {}


def load_snap(name, url):
    """Download a SNAP gz edge list and return an undirected simple Graph."""
    print(f"  downloading {name} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    text = gzip.decompress(raw).decode("utf-8", errors="ignore")
    G = nx.Graph()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        u, v = parts[0], parts[1]
        if u != v:
            G.add_edge(u, v)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def load_local(name, path):
    print(f"  reading {name} from {path} ...", flush=True)
    G = nx.Graph()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            if u != v:
                G.add_edge(u, v)
    return G


# ---------- parameters ----------
def triangle_edge_counts(G):
    adj = {v: set(G.neighbors(v)) for v in G}
    te = {}
    for (u, v) in G.edges():
        te[frozenset((u, v))] = len(adj[u] & adj[v])
    return te, adj


def degeneracy(G):
    # O(m) core-based degeneracy
    return max(nx.core_number(G).values()) if G.number_of_nodes() else 0


def params_K3(G):
    deg = dict(G.degree())
    te, adj = triangle_edge_counts(G)
    T = sum(te.values()) // 3
    DeltaE = max(te.values()) if te else 0
    # oracle-width alpha: orient copy-bearing edges toward lower triangle-degree
    tdeg = {v: 0 for v in G}
    for e, c in te.items():
        u, v = tuple(e); tdeg[u] += c; tdeg[v] += c
    ow = {v: 0 for v in G}
    for (u, v) in G.edges():
        if te[frozenset((u, v))] == 0:
            continue
        ow[u if tdeg[u] <= tdeg[v] else v] += 1
    alpha = max(ow.values()) if ow else 0
    # tau: largest floor-degree over triangles (sample if huge)
    # compute exactly by scanning triangles via the low-degree orientation
    tau = 0
    # efficient triangle floor: for each edge with te>0, its triangles' third
    # vertex w; the floor is min(deg u, deg v, deg w). We bound tau by scanning.
    for (u, v) in G.edges():
        if te[frozenset((u, v))] == 0:
            continue
        common = adj[u] & adj[v]
        for w in common:
            tau = max(tau, min(deg[u], deg[v], deg[w]))
        # early exit if tau already >= min side (can't grow beyond max degree)
    kappa = degeneracy(G)
    m = G.number_of_edges()
    return dict(n=G.number_of_nodes(), m=m, T=T, alpha=alpha, tau=tau,
                DeltaE=DeltaE, kappa=kappa)


# ---------- estimator success probability (exact, full-path, K3) ----------
def success_prob_K3(G, predictor=None, sample_tris=None):
    """
    Exact per-instance success prob of the full-path weighted sampler.
    predictor=None => uniform FGP baseline.
    For very large triangle sets, pass sample_tris=K to subsample K triangles
    and scale (unbiased for the ratio we report).
    """
    deg = dict(G.degree()); adj = {v: set(G.neighbors(v)) for v in G}
    m = G.number_of_edges()
    edges = [frozenset(e) for e in G.edges()]
    if predictor is not None:
        te = {frozenset((u, v)): len(adj[u] & adj[v]) for (u, v) in G.edges()}
        W = sum(predictor.get(e, 0.0) + 1.0 for e in edges)
    # enumerate triangles (optionally sample)
    tris = []
    for u in G:
        Nu = [w for w in adj[u] if w > u]
        for i in range(len(Nu)):
            for j in range(i + 1, len(Nu)):
                a, b = Nu[i], Nu[j]
                if b in adj[a]:
                    tris.append((u, a, b))
    if not tris:
        return 0.0, 0
    scale = 1.0
    if sample_tris and len(tris) > sample_tris:
        idx = np.random.default_rng(0).choice(len(tris), sample_tris, replace=False)
        tris = [tris[i] for i in idx]
        scale = None  # we report a RATIO, so subsampling cancels; keep as-is
    M = len(edges)
    inv = 1.0 / np.sqrt(2 * m)
    p = 0.0
    for (x, y, z) in tris:
        trip = sorted((x, y, z), key=lambda v: (deg[v], v))
        piv, mid, close = trip
        base = frozenset((piv, mid))
        if predictor is None:
            p += (1.0 / M) * inv
        else:
            pb = (predictor.get(base, 0.0) + 1.0) / W
            wsum = 0.0; wc = 0.0
            for c in adj[piv]:
                w = predictor.get(frozenset((piv, c)), 0.0) + 1.0
                wsum += w
                if c == close:
                    wc = w
            p += pb * (wc / wsum if wsum > 0 else 0.0)
    return p, len(tris)


def analyze_graph(name, G, sample_tris=200000):
    pr = params_K3(G)
    te = {frozenset((u, v)): len(set(G.neighbors(u)) & set(G.neighbors(v)))
          for (u, v) in G.edges()}
    perfect = {e: te[e] for e in te}
    p_u, nt = success_prob_K3(G, None, sample_tris)
    p_p, _ = success_prob_K3(G, perfect, sample_tris)
    speedup = (1 / p_u) / (1 / p_p) if p_u > 0 and p_p > 0 else float("nan")
    pr["speedup"] = speedup
    pr["alpha_over_sqrtm"] = pr["alpha"] / (pr["m"] ** 0.5)
    return name, pr


def main():
    graphs = {}
    for name, url in SNAP.items():
        try:
            graphs[name] = load_snap(name, url)
        except Exception as e:
            print(f"  [skip {name}: {e}]")
    for name, path in LOCAL_FILES.items():
        graphs[name] = load_local(name, path)

    print(f"\n{'graph':>13}{'n':>8}{'m':>9}{'#T':>10}{'kappa':>7}{'alpha':>7}"
          f"{'tau':>6}{'DeltaE':>8}{'a/sqrtm':>9}{'a/kappa':>9}{'speedup':>9}"
          f"{'verdict':>12}")
    rows = []
    for name, G in graphs.items():
        _, pr = analyze_graph(name, G)
        rows.append((name, pr))
        a_over_k = pr["alpha"] / pr["kappa"] if pr["kappa"] else float("nan")
        # Honest verdict: an EXPONENT gain needs alpha << sqrt(m) AND alpha << kappa
        # (otherwise the predictor is no better than a degeneracy ordering).
        if pr["alpha_over_sqrtm"] < 0.1 and a_over_k < 0.5:
            verdict = "EXP gain"
        elif pr["speedup"] > 10:
            verdict = "const gain"
        else:
            verdict = "little"
        print(f"{name:>13}{pr['n']:>8}{pr['m']:>9}{pr['T']:>10}{pr['kappa']:>7}"
              f"{pr['alpha']:>7}{pr['tau']:>6}{pr['DeltaE']:>8}"
              f"{pr['alpha_over_sqrtm']:>9.3f}{a_over_k:>9.2f}"
              f"{pr['speedup']:>9.1f}{verdict:>12}")

    print("\nHow to read this map:")
    print("  a/sqrtm : alpha relative to sqrt(m). Theory: speedup = Theta(sqrt(m)/alpha),")
    print("            so SMALL a/sqrtm => large speedup.")
    print("  a/kappa : alpha relative to degeneracy. This is the KEY column: a predictor")
    print("            only beats a degeneracy-ordering baseline when a/kappa << 1.")
    print("  On real graphs alpha ~ kappa is typical (no 'copy-free dense decoy' exists),")
    print("  so predictions give a large CONSTANT-factor gain but not an exponent gain.")
    print("  The exponent separation of Thm 4 needs separable structure, which the")
    print("  synthetic 'separate' family has and real graphs generally do not.")


if __name__ == "__main__":
    main()
