#!/usr/bin/env python3
"""
C4 on REAL graphs -- companion to real_graphs.py (which does K3).

CRITICAL practical difference: real graphs have ENORMOUS numbers of C4s
(facebook has ~10^9+), so we CANNOT enumerate them. Strategy:

  * #C4 total: computed exactly and cheaply via the wedge formula
        #C4 = (1/2) * sum over vertex pairs {x,y} of C(codeg(x,y), 2)
    using a codegree sweep -- O(sum deg^2) which is fine for these sizes.

  * per-edge C4 weights: also O(sum deg^2) via codegrees.

  * success probabilities: we do NOT enumerate C4s. Instead we compute the
    success probability ANALYTICALLY:
       uniform:  p_u = #C4_canonical / M^2      (canonical = 1 opposite pair)
       weighted: p_w = sum over C4s of q(e1)q(e2)
    The weighted sum is computed WITHOUT enumeration by noting that for the
    canonical opposite-pair decomposition,
       sum over C4s q(e1)q(e2)  =  (1/2) * sum over vertex-pairs {x,y}
                                    of [ sum over pairs of common nbrs ... ]
    which is still expensive. So instead we SAMPLE C4s uniformly (via the
    wedge distribution) and estimate the ratio p_w/p_u by importance sampling,
    which is exactly the speedup we want. Sampling K=200k C4s gives a tight
    estimate of the ratio (that's all we report).

Run in Colab:
    !python c4_real.py
    # or: import c4_real; c4_real.main()
"""
import gzip
import os
import sys
import urllib.request
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from alpha_exact import alpha_from_cb_edges   # exact alpha_H (Definition 3)

SNAP = {
    "ca-GrQc":     "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
    "ca-HepTh":    "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
    "email-Enron": "https://snap.stanford.edu/data/email-Enron.txt.gz",
    "facebook":    "https://snap.stanford.edu/data/facebook_combined.txt.gz",
}

SAMPLE_C4 = 200_000     # number of C4s to sample for the ratio estimate


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


def codegree_stats(G, adj):
    """
    Returns (total_C4, c4_per_edge) using codegrees.
    #C4 = 1/2 * sum_{pairs {x,y}} C(codeg(x,y), 2)
    per-edge weight for edge {u,v}: number of C4s containing it. We use the
    standard identity: C4s through edge {u,v} = sum over x in N(u)\\{v}
    of (codeg(x,v) - [x~v]) ... which is costly; we instead use a cheaper
    PROXY weight that the predictor would realistically produce:
        w(u,v) = sum over x in N(u) of (codeg(x,v) - 1)_+   / 2
    computed via a codegree dictionary restricted to pairs with codeg >= 2.
    """
    codeg = {}
    for u in G:
        Nu = list(adj[u])
        L = len(Nu)
        for i in range(L):
            for j in range(i + 1, L):
                a, b = Nu[i], Nu[j]
                key = (a, b) if a < b else (b, a)
                codeg[key] = codeg.get(key, 0) + 1
    total = 0
    for k, c in codeg.items():
        if c >= 2:
            total += c * (c - 1) // 2
    total //= 2
    # per-edge weight: for edge {u,v}, C4s through it correspond to picking
    # x in N(u), y in N(v), x!=v, y!=u, x~y. Equivalently sum over x in N(u)\{v}
    # of (codeg(x,v) minus adjustments). We approximate with the codegree sum,
    # which is what a practical predictor would learn.
    w = {}
    for (u, v) in G.edges():
        s = 0
        for x in adj[u]:
            if x == v:
                continue
            key = (x, v) if x < v else (v, x)
            c = codeg.get(key, 0)
            if c >= 1:
                s += c - 1
        w[frozenset((u, v))] = max(s, 0)
    return total, w, codeg


def alpha_C4(G, w):
    """
    EXACT oracle-width alpha_{C4} = pseudoarboricity of the C4-bearing subgraph
    (Definition 3), via max-flow. See src/alpha_exact.py. (Replaces the previous
    greedy-orientation upper bound, which could exceed kappa.)
    """
    cb_edges = [tuple(e) for e in w if w[e] > 0]
    kappa_hint = max(nx.core_number(G).values()) if G.number_of_edges() else 0
    return alpha_from_cb_edges(cb_edges, kappa_hint=kappa_hint)


def sample_c4s(G, adj, codeg, k, rng):
    """
    Sample C4s (approximately uniformly) by sampling a vertex pair {x,y}
    proportional to C(codeg,2), then two distinct common neighbours.
    Returns a list of (x, y, u, v) where u,v are the two common neighbours,
    i.e. the C4 is x-u-y-v-x. Opposite edge pair used as canonical: {x,u},{y,v}.
    """
    keys = [k_ for k_, c in codeg.items() if c >= 2]
    if not keys:
        return []
    weights = np.array([codeg[k_] * (codeg[k_] - 1) / 2 for k_ in keys], dtype=float)
    weights /= weights.sum()
    idx = rng.choice(len(keys), size=min(k, 500000), replace=True, p=weights)
    out = []
    for i in idx:
        x, y = keys[i]
        common = list(adj[x] & adj[y])
        if len(common) < 2:
            continue
        a, b = rng.choice(len(common), size=2, replace=False)
        u, v = common[a], common[b]
        out.append((x, y, u, v))
    return out


def speedup_C4(G, adj, codeg, w, rng, k=SAMPLE_C4):
    """
    Estimate speedup = (1/p_uniform) / (1/p_weighted) = p_weighted / p_uniform
    via importance sampling over C4s. For a sampled C4 with canonical opposite
    edges e1={x,u}, e2={y,v}:
        uniform contribution  : (1/M)^2
        weighted contribution : q(e1) * q(e2),  q(e) = (w(e)+1)/W
    The ratio of the SUMS over all C4s equals the ratio of the MEANS over a
    uniform sample of C4s -- but our sample is proportional to codegree pairs,
    which is uniform over C4s (each C4 has exactly one diagonal pair {x,y}
    with the other two vertices as common neighbours... actually each C4 has
    2 diagonal pairs, so the sampling is uniform up to a factor 2 that cancels
    in the ratio).
    """
    edges = list(G.edges())
    M = len(edges)
    W = sum(w.get(frozenset(e), 0) + 1.0 for e in edges)
    c4s = sample_c4s(G, adj, codeg, k, rng)
    if not c4s:
        return float("nan"), 0
    unif = (1.0 / M) ** 2
    ratios = []
    for (x, y, u, v) in c4s:
        e1 = frozenset((x, u))
        e2 = frozenset((y, v))
        q1 = (w.get(e1, 0) + 1.0) / W
        q2 = (w.get(e2, 0) + 1.0) / W
        ratios.append((q1 * q2) / unif)
    return float(np.mean(ratios)), len(c4s)


def main():
    rng = np.random.default_rng(0)
    print(f"{'graph':>13}{'m':>9}{'#C4':>14}{'kappa':>7}{'alpha':>8}"
          f"{'a/sqrtm':>9}{'a/kappa':>9}{'speedup':>10}{'verdict':>12}")
    for name, url in SNAP.items():
        try:
            G = load_snap(name, url)
        except Exception as e:
            print(f"  [skip {name}: {e}]")
            continue
        adj = {v: set(G.neighbors(v)) for v in G}
        m = G.number_of_edges()
        try:
            total, w, codeg = codegree_stats(G, adj)
        except MemoryError:
            print(f"  [skip {name}: too large for exact codegrees]")
            continue
        kappa = max(nx.core_number(G).values())
        al = alpha_C4(G, w)
        sp, ns = speedup_C4(G, adj, codeg, w, rng)
        a_sq = al / (m ** 0.5)
        a_k = al / kappa if kappa else float("nan")
        if a_sq < 0.1 and a_k < 0.5:
            verdict = "EXP gain"
        elif sp > 5:
            verdict = "const gain"
        else:
            verdict = "little"
        print(f"{name:>13}{m:>9}{total:>14}{kappa:>7}{al:>8}"
              f"{a_sq:>9.3f}{a_k:>9.2f}{sp:>10.1f}{verdict:>12}")

    print("\nCompare with the K3 table (real_graphs.py):")
    print("  For K3 we found alpha ~ kappa on all real graphs => constant-factor")
    print("  gains only. The question here is whether C4 behaves the same, or")
    print("  whether its higher rho (=2) yields a larger realized speedup.")


if __name__ == "__main__":
    main()
