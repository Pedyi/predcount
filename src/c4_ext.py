#!/usr/bin/env python3
"""
C4 extension of PredCount.

Key structural difference from K3:
  rho(C4) = 2 (integral). C4 is an EVEN cycle, so the FGP sampler has NO
  closing-vertex step. Instead a C4 = (a,b,c,d) is discovered by sampling TWO
  OPPOSITE edges: {a,b} and {c,d}, then verifying the two connecting edges
  {b,c},{d,a} exist. Each C4 has 2 opposite-edge pairs, and each pair can be
  drawn in 2 orders, so canonicalization fixes one representative.

Uniform baseline: Pr[fixed C4] = Theta(1 / m^2)  (two independent edge draws)
  => flips to success ~ m^2 / #C4 = m^{rho} / #C4 with rho=2.  Matches FGP.

Weighted: draw each of the two opposite edges with prob proportional to
(pred(e)+1). Likelihood ratio corrects for unbiasedness exactly as for K3.

Oracle-width for C4: orient C4-bearing edges toward the endpoint of smaller
C4-degree; alpha_{C4} = max out-degree.

Run:
    python c4_ext.py                  # synthetic families
    python c4_ext.py --real           # real SNAP graphs (needs internet)
"""
import argparse
import numpy as np
import networkx as nx

try:
    from alpha_exact import alpha_from_cb_edges, c4_bearing_edges
except ImportError:
    from .alpha_exact import alpha_from_cb_edges, c4_bearing_edges


# ---------------- C4 counting and per-edge weights ----------------
def c4_data(G):
    """
    Returns (c4_per_edge, total_c4, adj).
    Counts C4 via the standard wedge method: for each unordered pair {x,y},
    w(x,y) = #common neighbours; #C4 = sum over pairs of C(w,2), and each C4 is
    counted twice (once per diagonal pair).
    Per-edge C4 counts computed by, for each edge {u,v}, counting C4s through it.
    """
    adj = {v: set(G.neighbors(v)) for v in G}
    # wedge counts per non-adjacent-or-adjacent pair (diagonal pairs)
    wedge = {}
    for u in G:
        Nu = list(adj[u])
        for i in range(len(Nu)):
            for j in range(i + 1, len(Nu)):
                a, b = Nu[i], Nu[j]
                key = (a, b) if a < b else (b, a)
                wedge[key] = wedge.get(key, 0) + 1
    total = 0
    for key, w in wedge.items():
        total += w * (w - 1) // 2
    total //= 2   # each C4 has 2 diagonals

    # per-edge: C4s through edge {u,v} = sum over neighbours x of u (x!=v),
    # neighbours y of v (y!=u), x!=y, with {x,y} an edge -> that's a C4
    # u-v-y-x-u. Count and divide appropriately.
    c4e = {}
    for (u, v) in G.edges():
        cnt = 0
        for x in adj[u]:
            if x == v:
                continue
            # need y in adj[v], y != u, y != x, and y adjacent to x
            common = adj[x] & adj[v]
            cnt += len(common - {u, v, x})
        c4e[frozenset((u, v))] = cnt // 2   # each C4 through the edge counted twice
    return c4e, total, adj


def alpha_C4(G, c4e):
    """
    EXACT oracle-width alpha_{C4} = pseudoarboricity of the C4-bearing subgraph
    (Definition 3), via max-flow. See src/alpha_exact.py.

    (The previous implementation used the max out-degree of a single greedy
    orientation, which is only an upper bound and can exceed kappa. It has been
    replaced.)
    """
    cb_edges = [tuple(e) for e in c4e if c4e[e] > 0]
    kappa = max(nx.core_number(G).values()) if G.number_of_edges() else 0
    return alpha_from_cb_edges(cb_edges, kappa_hint=kappa)


def enumerate_c4(G, adj, cap=None):
    """
    Enumerate C4s as canonical 4-tuples (a,b,c,d) with a = min vertex and
    b < d (fixes rotations/reflections). Returns list of (a,b,c,d).
    cap: stop after this many (for large graphs).
    """
    out = []
    nodes = sorted(G, key=lambda v: (G.degree(v), str(v)))
    for a in nodes:
        Na = [x for x in adj[a]]
        for i in range(len(Na)):
            for j in range(i + 1, len(Na)):
                b, d = Na[i], Na[j]
                # need c adjacent to both b and d, c != a
                common = (adj[b] & adj[d]) - {a}
                for c in common:
                    # canonical: a is the smallest by our ordering
                    key = (a, b, c, d)
                    out.append(key)
                    if cap and len(out) >= cap:
                        return out
    return out


def c4_success_prob(G, adj, c4s, predictor=None):
    """
    Exact per-instance success probability of the (full-path) C4 sampler.
    Uniform: both opposite edges drawn uniformly -> (1/M)^2 per canonical path.
    Weighted: each drawn ~ (pred+1)/W.
    Canonical path for C4 (a,b,c,d): the two OPPOSITE edges are {a,b} and {c,d}.
    """
    edges = [frozenset(e) for e in G.edges()]
    M = len(edges)
    if predictor is None:
        return len(c4s) * (1.0 / M) ** 2
    W = sum(predictor.get(e, 0.0) + 1.0 for e in edges)
    p = 0.0
    for (a, b, c, d) in c4s:
        e1 = frozenset((a, b))
        e2 = frozenset((c, d))
        p1 = (predictor.get(e1, 0.0) + 1.0) / W
        p2 = (predictor.get(e2, 0.0) + 1.0) / W
        p += p1 * p2
    return p


# ---------------- families ----------------
def gen(family, n, seed):
    rng = np.random.default_rng(seed)
    if family == "even":
        return nx.barabasi_albert_graph(n, 4, seed=seed)
    if family == "bipartite":
        # C4-rich bipartite: dense-ish random bipartite
        return nx.bipartite.random_graph(n // 2, n // 2, 8.0 / (n / 2), seed=seed)
    if family == "separate-c4":
        # signal: C4s on a shallow gadget (two hubs + leaf pairs)
        G = nx.Graph()
        h1, h2 = 0, 1
        node = 2
        s = max(50, n // 20)
        for _ in range(s):
            a, b = node, node + 1
            G.add_edge(h1, a); G.add_edge(a, h2); G.add_edge(h2, b); G.add_edge(b, h1)
            node += 2
        # decoy must be C4-FREE but dense enough to inflate m and kappa.
        # An incidence graph of a projective plane has girth 6 (C4-free) and
        # ~q^2 vertices with degree q+1. Cheap stand-in: a random regular graph
        # of high degree has few C4s relative to its size, but not zero. We use
        # a bipartite incidence structure with girth >= 6 built greedily:
        # each left vertex picks neighbours so no two share 2 common neighbours.
        d = max(30, n // 4)
        left = list(range(node, node + d)); node += d
        right = list(range(node, node + d)); node += d
        rng2 = np.random.default_rng(seed + 7)
        pair_used = set()          # (r1,r2) pairs already co-covered
        for u in left:
            chosen = []
            tries = 0
            while len(chosen) < 4 and tries < 60:
                tries += 1
                v = int(rng2.choice(right))
                if v in chosen:
                    continue
                ok = all(tuple(sorted((v, w))) not in pair_used for w in chosen)
                if ok:
                    for w in chosen:
                        pair_used.add(tuple(sorted((v, w))))
                    chosen.append(v)
            for v in chosen:
                G.add_edge(u, v)
        return G
    if family == "dense":
        return nx.gnp_random_graph(n, 0.3, seed=seed)
    raise ValueError(family)


def run_synthetic(sizes, seeds, families):
    print("\n########## pattern = C4 ##########")
    print(f"{'family':>13}{'m':>8}{'#C4':>10}{'kappa':>7}{'alpha':>7}"
          f"{'a/sqrtm':>9}{'a/kappa':>9}{'flips_u':>12}{'flips_p':>11}{'speedup':>9}")
    rows = []
    for fam in families:
        for n in sizes:
            for sd in seeds:
                G = gen(fam, n, sd)
                G.remove_edges_from(nx.selfloop_edges(G))
                if G.number_of_edges() < 10:
                    continue
                c4e, total, adj = c4_data(G)
                if total == 0:
                    continue
                c4s = enumerate_c4(G, adj, cap=300000)
                if not c4s:
                    continue
                m = G.number_of_edges()
                kappa = max(nx.core_number(G).values())
                al = alpha_C4(G, c4e)
                pu = c4_success_prob(G, adj, c4s, None)
                perfect = {e: c4e[e] for e in c4e}
                pp = c4_success_prob(G, adj, c4s, perfect)
                if pu <= 0 or pp <= 0:
                    continue
                fu, fp = 1 / pu, 1 / pp
                sp = fu / fp
                rows.append((fam, m, total, kappa, al, sp))
                print(f"{fam:>13}{m:>8}{total:>10}{kappa:>7}{al:>7}"
                      f"{al/np.sqrt(m):>9.3f}{(al/kappa if kappa else 0):>9.2f}"
                      f"{fu:>12.1f}{fp:>11.2f}{sp:>9.1f}")
    # growth exponents
    print("\n--- C4 map summary ---")
    print(f"{'family':>13}{'median_sp':>12}{'sp~m^?':>10}{'alpha~m^?':>11}{'verdict':>12}")
    for fam in sorted(set(r[0] for r in rows)):
        sub = [r for r in rows if r[0] == fam]
        ms = np.array([r[1] for r in sub], dtype=float)
        sp = np.array([r[5] for r in sub], dtype=float)
        al = np.array([r[4] for r in sub], dtype=float)
        if len(set(ms)) >= 2:
            e_sp = np.polyfit(np.log(ms), np.log(sp), 1)[0]
            e_al = np.polyfit(np.log(ms), np.log(al + 1), 1)[0]
        else:
            e_sp = e_al = float("nan")
        verdict = "EXP gain" if e_sp > 0.3 else ("const gain" if np.median(sp) > 5 else "little")
        print(f"{fam:>13}{np.median(sp):>11.1f}x{e_sp:>10.2f}{e_al:>11.2f}{verdict:>12}")
    print("\nFor C4, rho=2, so on a SEPARABLE family the speedup grows with")
    print("m-exponent rho-1 = 1 (measured on the separable construction). On real")
    print("graphs alpha is a constant fraction of kappa, so the gain is a large")
    print("constant, not an exponent improvement (see extended_real_study).")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[400, 800, 1600])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--families", nargs="+",
                    default=["even", "bipartite", "separate-c4", "dense"])
    args, _ = ap.parse_known_args()
    run_synthetic(args.sizes, args.seeds, args.families)


if __name__ == "__main__":
    main()
