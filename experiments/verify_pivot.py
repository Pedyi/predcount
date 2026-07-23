"""
CRITICAL CHECK for Lemma (weighted success prob): is the pivot's total incident
weight D_x = sum_{c in N(x)} (t({x,c})+1) actually O(alpha) on copy-bearing
pivots under the low-t orientation sigma*?

This is the weakest link in the consistency proof. If D_x is NOT O(alpha) (e.g.
if a high-degree hub is a pivot), the success-probability lower bound changes.

We compute, over the SEPARATE family (the one that drives the theorem) and an
EVEN family, for every triangle its canonical pivot x, and measure D_x. We
report max/mean D_x over copy-bearing pivots vs alpha, and how D_x scales with m.

If D_x stays O(alpha)=O(1) on SEPARATE, the lemma's constant is safe. If instead
D_x tracks deg(pivot), we must REWRITE the lemma with D_x = O(deg_pivot + alpha)
and re-derive p_succ -- which would change the theorem's alpha-dependence.
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

def pivot_weights(G):
    deg=dict(G.degree()); adj={v:set(G.neighbors(v)) for v in G}
    te={frozenset((u,v)):len(adj[u]&adj[v]) for (u,v) in G.edges()}
    # alpha via low-t orientation
    tdeg={v:0 for v in G}
    for e,c in te.items():
        u,v=tuple(e); tdeg[u]+=c; tdeg[v]+=c
    ow={v:0 for v in G}
    for (u,v) in G.edges():
        if te[frozenset((u,v))]==0: continue
        ow[u if tdeg[u]<=tdeg[v] else v]+=1
    alpha=max(ow.values())
    # enumerate triangles, canonical pivot = min (deg,id)
    Dx_list=[]; pivot_is_hub=[]
    tris=[]
    for u in G:
        Nu=[w for w in adj[u] if w>u]
        for i in range(len(Nu)):
            for j in range(i+1,len(Nu)):
                a,b=Nu[i],Nu[j]
                if b in adj[a]: tris.append((u,a,b))
    for (x,y,z) in tris:
        piv=sorted((x,y,z), key=lambda v:(deg[v],v))[0]
        Dx=sum((te[frozenset((piv,c))] if frozenset((piv,c)) in te else 0)+1
                for c in adj[piv])
        Dx_list.append(Dx)
    Dx=np.array(Dx_list)
    return alpha, Dx.max(), Dx.mean(), len(tris)

print("=== SEPARATE ===")
print(f"{'m':>8}{'alpha':>7}{'maxDx':>8}{'meanDx':>9}{'maxDx/alpha':>12}")
sep=[]
for s in [200,400,800,1600,3200]:
    G=build_separate(s); m=G.number_of_edges()
    a,mx,mn,T=pivot_weights(G); sep.append((m,a,mx,mn))
    print(f"{m:>8}{a:>7}{mx:>8.0f}{mn:>9.2f}{mx/a:>12.2f}")
ms=np.array([r[0] for r in sep])
print(f"  maxDx ~ m^{np.polyfit(np.log(ms),np.log([r[2] for r in sep]),1)[0]:.3f}")
print(f"  meanDx ~ m^{np.polyfit(np.log(ms),np.log([r[3] for r in sep]),1)[0]:.3f}")

print("\n=== EVEN ===")
print(f"{'m':>8}{'alpha':>7}{'maxDx':>8}{'meanDx':>9}{'maxDx/alpha':>12}")
ev=[]
for n in [1000,2000,4000,8000]:
    G=build_even(n); m=G.number_of_edges()
    a,mx,mn,T=pivot_weights(G); ev.append((m,a,mx,mn))
    print(f"{m:>8}{a:>7}{mx:>8.0f}{mn:>9.2f}{mx/a:>12.2f}")
ms=np.array([r[0] for r in ev])
print(f"  maxDx ~ m^{np.polyfit(np.log(ms),np.log([r[2] for r in ev]),1)[0]:.3f}")
print(f"  meanDx ~ m^{np.polyfit(np.log(ms),np.log([r[3] for r in ev]),1)[0]:.3f}")

print("\nKEY QUESTION: is meanDx = O(alpha)?  (mean matters: p_succ sums over")
print("triangles, so the AVERAGE pivot weight enters, not the max.) If meanDx is")
print("~constant on SEPARATE, the lemma holds as written with D_x -> meanDx.")
