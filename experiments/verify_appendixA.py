"""
APPENDIX A groundwork: find the PRECISE condition under which delta* = O(alpha).

Claim to pin down: delta* = max over canonical pivots x of D_x, where
D_x = deg(x) + sum_{c~x} t({x,c}).  Canonical pivot = min-(deg,id) vertex of the
triangle. So delta* is controlled by the degree of the LOWEST-degree vertex of
each triangle. Define:
   d_min(C) = min degree over the 3 vertices of triangle C.
   Dmax_triangles = max_C d_min(C)   [the largest 'floor' degree across triangles]
Then every canonical pivot has degree <= Dmax_triangles, so
   delta* <= Dmax_triangles + (triangle incidence of pivot).
The triangle incidence of a pivot is <= alpha by sigma* only if the pivot is the
low-t endpoint -- but canonical pivot is low-DEGREE, not low-t. So we need to
relate low-degree to low-t.

PROPOSED separability condition (S):
   For every triangle C, its min-degree vertex also has triangle-incidence O(alpha).
Equivalently: the vertices that are 'floors' of triangles are themselves shallow.

We test whether condition (S) => delta* = O(alpha), and measure the quantity
   R = delta* / (Dmax_triangles + alpha)
which should be Theta(1) universally (that's just the definition), AND separately
whether on 'separate' the binding term is alpha (good) vs deg on 'even' (bad).

The real deliverable: identify the clean parameter to put in the theorem.
Candidate: tau := max_C d_min(C) = max over triangles of the min-degree vertex.
Then delta* = O(tau) ALWAYS (pivot degree <= tau, and incidence <= tau too since
incidence <= degree). And separability = "tau = O(alpha)".
Let's verify delta* = Theta(tau) on both families and tau's scaling.
"""
import numpy as np
import networkx as nx

def build_separate(s):
    d=int(round(s**0.5)); G=nx.Graph(); hub=0; node=1
    for _ in range(s):
        a,b=node,node+1
        G.add_edge(hub,a);G.add_edge(hub,b);G.add_edge(a,b);node+=2
    L=list(range(node,node+d));R=list(range(node+d,node+2*d))
    for u in L:
        for v in R: G.add_edge(u,v)
    return G

def build_even(n): return nx.barabasi_albert_graph(n,4,seed=1)

def measure(G):
    deg=dict(G.degree()); adj={v:set(G.neighbors(v)) for v in G}
    te={frozenset((u,v)):len(adj[u]&adj[v]) for (u,v) in G.edges()}
    tdeg={v:0 for v in G}
    for e,c in te.items():
        u,v=tuple(e);tdeg[u]+=c;tdeg[v]+=c
    ow={v:0 for v in G}
    for (u,v) in G.edges():
        if te[frozenset((u,v))]==0: continue
        ow[u if tdeg[u]<=tdeg[v] else v]+=1
    alpha=max(ow.values())
    tris=[]
    for u in G:
        Nu=[w for w in adj[u] if w>u]
        for i in range(len(Nu)):
            for j in range(i+1,len(Nu)):
                a,b=Nu[i],Nu[j]
                if b in adj[a]: tris.append((u,a,b))
    tau=0; deltastar=0
    for (x,y,z) in tris:
        dmin=min(deg[x],deg[y],deg[z])
        tau=max(tau,dmin)
        piv=sorted((x,y,z),key=lambda v:(deg[v],v))[0]
        Dx=deg[piv]+sum(te.get(frozenset((piv,c)),0) for c in adj[piv])
        deltastar=max(deltastar,Dx)
    return alpha,tau,deltastar

print("=== SEPARATE ===")
print(f"{'m':>8}{'alpha':>7}{'tau':>6}{'delta*':>8}{'delta*/tau':>11}{'tau/alpha':>10}")
sep=[]
for s in [200,400,800,1600,3200]:
    G=build_separate(s); m=G.number_of_edges()
    a,tau,ds=measure(G); sep.append((m,a,tau,ds))
    print(f"{m:>8}{a:>7}{tau:>6}{ds:>8}{ds/tau:>11.2f}{tau/a:>10.2f}")

print("\n=== EVEN ===")
print(f"{'m':>8}{'alpha':>7}{'tau':>6}{'delta*':>8}{'delta*/tau':>11}{'tau/alpha':>10}")
for n in [1000,2000,4000,8000]:
    G=build_even(n); m=G.number_of_edges()
    a,tau,ds=measure(G)
    print(f"{m:>8}{a:>7}{tau:>6}{ds:>8}{ds/tau:>11.2f}{tau/a:>10.2f}")

print("\nConclusion to formalize:")
print(" - delta* = Theta(tau) on both families => tau is the clean parameter.")
print(" - tau = max over triangles of its MIN-degree vertex.")
print(" - Separability condition (S): tau = O(alpha).")
print(" - Under (S): delta* = O(alpha), so Lemma's p_succ = Theta(#T/(m*alpha)).")
