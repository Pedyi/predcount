#!/usr/bin/env python3
"""
Comparison against reimplemented competing methods (addresses reviewer point 4).

IMPORTANT DISCLAIMER
--------------------
We do NOT have the original authors' code. What follows is our own
reimplementation of the ALGORITHMIC IDEA of two competing approaches, written
from their published descriptions. Differences in constants, tie-breaking and
engineering choices are therefore possible, and the comparison should be read as
indicative of the mechanisms, not as a benchmark of the authors' artifacts. We
state this explicitly in the paper.

METHODS COMPARED (all for triangle counting, K3)
------------------------------------------------
  FGP        : the prediction-free Fichtenberger-Peng baseline.

  HEAVY-SEP  : the "heavy-edge separation" idea used by Chen, Eden, Indyk,
               Woodruff et al. (ICLR 2022). A heavy-edge oracle identifies the
               top-tau edges by predicted triangle count; those are stored
               exactly and their triangles counted deterministically, while the
               remaining light edges are handled by uniform sampling. Space is
               tau (for the stored heavy edges) plus the sampling budget.

  HEAVY-DROP : the "heaviness oracle" idea of Luderssen, Neumann and Peng
               (2026). Edges whose predicted copy-count exceeds a threshold are
               EXCLUDED from the sampling estimator (their contribution is
               computed separately), which reduces the estimator's variance.

  PREDCOUNT  : ours -- full-path importance weighting with likelihood-ratio
               correction and a 1/2 robustness mixture.

METRIC
------
For a fair comparison across mechanisms that spend space differently, we report
the TOTAL SPACE (in edge-slots) needed to reach a target relative error of 10%,
counting both stored edges and sampler instances. Lower is better.

Usage:
    python compare_methods.py
    python compare_methods.py --sizes 2000 4000 8000
"""
import argparse
import numpy as np
import networkx as nx


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
    raise ValueError(family)


def prep(G):
    deg = dict(G.degree())
    adj = {v: set(G.neighbors(v)) for v in G}
    te = {frozenset((u, v)): len(adj[u] & adj[v]) for (u, v) in G.edges()}
    T = sum(te.values()) // 3
    return deg, adj, te, T


def canonical_pivot(tri, deg):
    return sorted(tri, key=lambda v: (deg[v], v))


def enumerate_tris(G, adj):
    out = []
    for u in G:
        Nu = [w for w in adj[u] if w > u]
        for i in range(len(Nu)):
            for j in range(i + 1, len(Nu)):
                a, b = Nu[i], Nu[j]
                if b in adj[a]:
                    out.append((u, a, b))
    return out


# ----------------------------------------------------------------------
# space-to-target-error for each method
# ----------------------------------------------------------------------
def space_fgp(G, deg, adj, tris, eps=0.1):
    """Baseline: k instances at uniform success prob; space = k."""
    m = G.number_of_edges(); M = m
    inv = 1.0 / np.sqrt(2 * m)
    p = len(tris) * (1.0 / M) * inv
    if p <= 0:
        return np.inf
    return 1.0 / (p * eps ** 2)


def space_predcount(G, deg, adj, te, tris, eps=0.1):
    """Ours: full-path weighting; space = k (no stored edges)."""
    m = G.number_of_edges()
    edges = [frozenset(e) for e in G.edges()]
    W = sum(te.get(e, 0) + 1.0 for e in edges)
    p = 0.0
    for (x, y, z) in tris:
        piv, mid, close = canonical_pivot((x, y, z), deg)
        pb = (te.get(frozenset((piv, mid)), 0) + 1.0) / W
        wsum = wc = 0.0
        for c in adj[piv]:
            w = te.get(frozenset((piv, c)), 0) + 1.0
            wsum += w
            if c == close:
                wc = w
        p += pb * (wc / wsum if wsum > 0 else 0.0)
    if p <= 0:
        return np.inf
    return 1.0 / (p * eps ** 2)


def space_heavy_sep(G, deg, adj, te, tris, eps=0.1, tau_frac=0.01):
    """
    Chen et al. style: store the top tau edges exactly (space tau), count their
    triangles deterministically, and sample uniformly for the rest.
    Total space = tau + sampling budget for the residual triangles.
    We sweep tau_frac and report the best (most favourable to the competitor).
    """
    m = G.number_of_edges()
    edges = sorted(te.items(), key=lambda kv: -kv[1])
    best = np.inf
    # sweep the storage fraction widely, including regimes where storing the
    # heavy edges covers ALL triangles (then no sampling is needed at all).
    for tf in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
        tau = max(1, int(tf * m))
        heavy = set(e for e, _ in edges[:tau])
        covered = 0
        for (x, y, z) in tris:
            es = [frozenset((x, y)), frozenset((x, z)), frozenset((y, z))]
            if any(e in heavy for e in es):
                covered += 1
        residual = len(tris) - covered
        if residual == 0:
            total = tau
        else:
            inv = 1.0 / np.sqrt(2 * m)
            p = residual * (1.0 / m) * inv
            total = tau + (1.0 / (p * eps ** 2) if p > 0 else np.inf)
        best = min(best, total)
    return best


def space_heavy_drop(G, deg, adj, te, tris, eps=0.1):
    """
    LNP style: exclude edges whose heaviness exceeds a threshold from the
    sampling estimator (handled separately, cost = number of such edges), which
    lowers the variance of the remainder. Sweep the threshold, report the best.
    """
    m = G.number_of_edges()
    vals = sorted(set(te.values()), reverse=True)
    best = np.inf
    # exhaustive threshold sweep over all distinct heaviness values
    for thr in vals + [0]:
        dropped = set(e for e, c in te.items() if c > thr)
        kept_tris = []
        for (x, y, z) in tris:
            es = [frozenset((x, y)), frozenset((x, z)), frozenset((y, z))]
            if not any(e in dropped for e in es):
                kept_tris.append((x, y, z))
        inv = 1.0 / np.sqrt(2 * m)
        p = len(kept_tris) * (1.0 / m) * inv
        samp = (1.0 / (p * eps ** 2)) if p > 0 else 0.0
        total = len(dropped) + samp
        best = min(best, total)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[2000, 4000, 8000])
    ap.add_argument("--families", nargs="+",
                    default=["even", "skewed", "mixed", "separate"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--eps", type=float, default=0.1)
    args, _ = ap.parse_known_args()

    print("NOTE: HEAVY-SEP and HEAVY-DROP are OUR REIMPLEMENTATIONS of the")
    print("      published algorithmic ideas, not the authors' code.\n")
    print("Space (edge-slots) to reach 10% relative error; lower is better.")
    print(f"{'family':>9}{'m':>8}{'#T':>8}{'FGP':>12}{'HEAVY-SEP':>12}"
          f"{'HEAVY-DROP':>12}{'PREDCOUNT':>12}{'ours/best':>11}")
    agg = {}
    for fam in args.families:
        for n in args.sizes:
            fg = hs = hd = pc = 0.0
            cnt = 0
            for sd in args.seeds:
                G = gen(fam, n, sd)
                deg, adj, te, T = prep(G)
                if T == 0:
                    continue
                tris = enumerate_tris(G, adj)
                m = G.number_of_edges()
                a = space_fgp(G, deg, adj, tris, args.eps)
                b = space_heavy_sep(G, deg, adj, te, tris, args.eps)
                c = space_heavy_drop(G, deg, adj, te, tris, args.eps)
                d = space_predcount(G, deg, adj, te, tris, args.eps)
                fg += a; hs += b; hd += c; pc += d; cnt += 1
                last_m, last_T = m, T
            if cnt == 0:
                continue
            fg, hs, hd, pc = fg / cnt, hs / cnt, hd / cnt, pc / cnt
            best_other = min(fg, hs, hd)
            print(f"{fam:>9}{last_m:>8}{last_T:>8}{fg:>12.0f}{hs:>12.0f}"
                  f"{hd:>12.0f}{pc:>12.0f}{pc/best_other:>11.2f}")
            agg.setdefault(fam, []).append((pc, best_other))

    print("\nSummary (ratio ours / best competitor; <1 means we win):")
    for fam, v in agg.items():
        r = np.mean([p / b for p, b in v])
        print(f"  {fam:>9}: {r:.2f}")
    print("\nReading: HEAVY-SEP pays space to STORE heavy edges but then counts")
    print("their triangles for free; HEAVY-DROP pays to exclude them. Both are")
    print("pattern-specific (triangles). PredCount stores nothing and generalizes")
    print("to any H, so a ratio near 1 means we match specialized methods while")
    print("being general; a ratio below 1 means we also win outright.")


if __name__ == "__main__":
    main()
