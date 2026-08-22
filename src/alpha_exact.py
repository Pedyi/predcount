#!/usr/bin/env python3
"""
Exact oracle-width alpha_H (Definition 3).

alpha_H(G) is the least, over orientations of the COPY-BEARING edges, of the
maximum out-degree. This equals the pseudoarboricity of the copy-bearing
subgraph,

    alpha_H = ceil( max_{S subseteq V}  |E(S)| / |S| ),

which we compute EXACTLY by an integer binary search over max-flow feasibility
(the standard Goldberg maximum-density-subgraph reduction). Only ~log2(kappa)
max-flow calls are needed, and the result is guaranteed to satisfy
alpha_H <= kappa (degeneracy).

IMPORTANT (why this module exists):
    An earlier version of the code estimated alpha_H by the max out-degree of a
    single GREEDY orientation (orient each edge toward its lower-weight
    endpoint). That is only an UPPER BOUND on alpha_H and can exceed kappa,
    which is impossible for the true parameter (Definition 3 guarantees
    alpha_H <= kappa). All alpha computations now route through this module so
    that the reported numbers are the exact parameter and are internally
    consistent with the paper.
"""
import networkx as nx

__all__ = ["alpha_from_cb_edges", "k3_bearing_edges", "c4_bearing_edges"]


def alpha_from_cb_edges(cb_edges, kappa_hint=None):
    """
    Exact alpha_H (Definition 3) from the list of copy-bearing edges.

    Parameters
    ----------
    cb_edges : iterable of (u, v)
        The copy-bearing edges (edges lying on at least one copy of H).
    kappa_hint : int, optional
        Degeneracy, used only as a starting upper bound for the search. Passing
        it makes the binary search start tight; the result is identical without
        it. The returned value is asserted to be <= kappa_hint when given.

    Returns
    -------
    int
        The exact pseudoarboricity of the copy-bearing subgraph, i.e. alpha_H.
    """
    H = nx.Graph()
    H.add_edges_from(cb_edges)
    if H.number_of_edges() == 0:
        return 0
    nodes = list(H.nodes())
    m = H.number_of_edges()
    deg = dict(H.degree())
    BIG = float(m + 1)

    def orientable_within(d):
        # True iff every subgraph S obeys |E(S)| <= d*|S|  (equivalently alpha <= d).
        F = nx.DiGraph()
        S, T = "__s__", "__t__"
        for u in nodes:
            F.add_edge(S, u, capacity=BIG)
            F.add_edge(u, T, capacity=max(BIG + 2.0 * d - deg[u], 0.0))
        for (u, v) in H.edges():
            F.add_edge(u, v, capacity=1.0)
            F.add_edge(v, u, capacity=1.0)
        cut, _ = nx.minimum_cut(F, S, T)
        return cut >= BIG * len(nodes) - 1e-6

    hi = kappa_hint if kappa_hint else m
    while not orientable_within(hi):
        hi *= 2
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if orientable_within(mid):
            hi = mid
        else:
            lo = mid + 1
    if kappa_hint is not None:
        assert lo <= kappa_hint, (
            f"alpha={lo} > kappa={kappa_hint}: impossible for the true "
            f"parameter (Definition 3). Check the copy-bearing edge set."
        )
    return lo


def k3_bearing_edges(G, adj=None):
    """Return the triangle-bearing edges of G (edges on >= 1 triangle)."""
    if adj is None:
        adj = {v: set(G.neighbors(v)) for v in G}
    return [(u, v) for (u, v) in G.edges() if len(adj[u] & adj[v]) > 0]


def c4_bearing_edges(G, adj=None):
    """
    Return the C4-bearing edges of G (edges lying on >= 1 four-cycle).

    An edge {u,v} is C4-bearing iff there exist x in N(u)\\{v} and
    y in N(v)\\{u} with x != y and {x,y} an edge (the 4-cycle u-v-y-x-u).
    """
    if adj is None:
        adj = {v: set(G.neighbors(v)) for v in G}
    out = []
    for (u, v) in G.edges():
        on = False
        for x in adj[u]:
            if x == v:
                continue
            # y adjacent to both x and v, y != u, y != x  => 4-cycle u-v-y-x-u
            if (adj[x] & adj[v]) - {u, v, x}:
                on = True
                break
        if on:
            out.append((u, v))
    return out


if __name__ == "__main__":
    # self-test: alpha <= kappa on a few graphs, and matches brute force on tiny ones
    import itertools

    def brute_alpha(cb_edges):
        H = nx.Graph(); H.add_edges_from(cb_edges)
        if H.number_of_edges() == 0:
            return 0
        best = max(1, ceil_density(H))
        return best

    def ceil_density(H):
        import math
        best = 0.0
        nodes = list(H.nodes())
        # exact max-density over all subsets is exponential; only for tiny graphs
        for r in range(1, len(nodes) + 1):
            for S in itertools.combinations(nodes, r):
                Sset = set(S)
                e = sum(1 for u, v in H.edges() if u in Sset and v in Sset)
                best = max(best, e / r)
        return math.ceil(best)

    for G in [nx.karate_club_graph(),
              nx.gnp_random_graph(30, 0.3, seed=1),
              nx.barabasi_albert_graph(40, 3, seed=2)]:
        adj = {v: set(G.neighbors(v)) for v in G}
        kap = max(nx.core_number(G).values())
        a3 = alpha_from_cb_edges(k3_bearing_edges(G, adj), kap)
        a4 = alpha_from_cb_edges(c4_bearing_edges(G, adj), kap)
        assert a3 <= kap and a4 <= kap
        print(f"n={G.number_of_nodes():3d} kappa={kap:3d}  alpha_K3={a3:3d}  alpha_C4={a4:3d}  (both <= kappa OK)")
    # brute-force check on a tiny graph
    Gt = nx.gnp_random_graph(9, 0.5, seed=5)
    adj = {v: set(Gt.neighbors(v)) for v in Gt}
    cb = k3_bearing_edges(Gt, adj)
    assert alpha_from_cb_edges(cb) == brute_alpha(cb), "brute-force mismatch"
    print("brute-force check passed.")
