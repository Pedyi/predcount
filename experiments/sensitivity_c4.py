#!/usr/bin/env python3
"""
Sensitivity analysis for the C4 predictor approximation (reviewer point 5).

For C4 on large graphs we used a codegree-based PROXY for per-edge copy counts
because exact counts are infeasible. The reviewer rightly asks whether this
methodological inconsistency drives the one anomalous result (email-Enron).

Here we compute BOTH the exact per-edge C4 counts and the codegree proxy on
graphs small enough for the exact computation, and measure how much the speedup
changes. If the gap is small and stable, the proxy is a sound substitute.
"""
import numpy as np, networkx as nx

def exact_c4_per_edge(G, adj):
    """Exact #C4 through each edge. O(sum deg^2 * deg) -- small graphs only."""
    w = {}
    for (u, v) in G.edges():
        cnt = 0
        for x in adj[u]:
            if x == v: continue
            for y in adj[v]:
                if y in (u, x): continue
                if y in adj[x]:
                    cnt += 1
        w[frozenset((u, v))] = cnt // 2
    return w

def proxy_c4_per_edge(G, adj):
    """Codegree-based proxy, as used at scale."""
    codeg = {}
    for u in G:
        Nu = list(adj[u])
        for i in range(len(Nu)):
            for j in range(i+1, len(Nu)):
                a, b = Nu[i], Nu[j]
                k = (a, b) if a < b else (b, a)
                codeg[k] = codeg.get(k, 0) + 1
    w = {}
    for (u, v) in G.edges():
        s = 0
        for x in adj[u]:
            if x == v: continue
            k = (x, v) if x < v else (v, x)
            c = codeg.get(k, 0)
            if c >= 1: s += c - 1
        w[frozenset((u, v))] = max(s, 0)
    return w

def c4_speedup(G, adj, w):
    """Speedup of weighted vs uniform opposite-edge sampling for C4."""
    edges = [frozenset(e) for e in G.edges()]
    M = len(edges)
    W = sum(w.get(e, 0) + 1.0 for e in edges)
    # enumerate C4 canonically: a<b<... via diagonal pairs
    out = 0.0; unif = 0.0
    seen = {}
    for u in G:
        Nu = list(adj[u])
        for i in range(len(Nu)):
            for j in range(i+1, len(Nu)):
                a, b = Nu[i], Nu[j]
                k = (a, b) if a < b else (b, a)
                seen.setdefault(k, []).append(u)
    for (x, y), commons in seen.items():
        if len(commons) < 2: continue
        for i in range(len(commons)):
            for j in range(i+1, len(commons)):
                u, v = commons[i], commons[j]
                e1 = frozenset((x, u)); e2 = frozenset((y, v))
                unif += (1.0/M)**2
                out += ((w.get(e1,0)+1.0)/W) * ((w.get(e2,0)+1.0)/W)
    return (out/unif) if unif > 0 else float('nan')

print(f"{'graph':>18}{'m':>7}{'sp_exact':>11}{'sp_proxy':>11}{'rel.diff':>10}")
tests = {
 "BA(400,4)":        nx.barabasi_albert_graph(400, 4, seed=1),
 "BA(600,5)":        nx.barabasi_albert_graph(600, 5, seed=2),
 "PLC(400,3,.4)":    nx.powerlaw_cluster_graph(400, 3, 0.4, seed=3),
 "PLC(500,5,.3)":    nx.powerlaw_cluster_graph(500, 5, 0.3, seed=4),
 "bipartite(200,.03)": nx.bipartite.random_graph(200, 200, 0.03, seed=5),
}
diffs=[]
for name, G in tests.items():
    G.remove_edges_from(nx.selfloop_edges(G))
    adj = {v:set(G.neighbors(v)) for v in G}
    we = exact_c4_per_edge(G, adj)
    wp = proxy_c4_per_edge(G, adj)
    se = c4_speedup(G, adj, we)
    sp = c4_speedup(G, adj, wp)
    d = (sp-se)/se if se>0 else float('nan')
    diffs.append(d)
    print(f"{name:>18}{G.number_of_edges():>7}{se:>11.1f}{sp:>11.1f}{d:>9.1%}")
d=np.array([x for x in diffs if np.isfinite(x)])
print(f"\nProxy vs exact: mean relative difference {d.mean():+.1%}, "
      f"range [{d.min():+.1%}, {d.max():+.1%}]")
print("\nIf the proxy systematically UNDER-states the speedup, the reported")
print("email-Enron C4 number is a conservative lower bound, not an anomaly.")
