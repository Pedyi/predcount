"""
CORRECTED separation: we need #T to grow so the honest gap
   oracle cost  Theta(m * alpha / #T)   [alpha=O(1)]  =  Theta(m/#T)
   lower bound  Omega(m / sqrt(#T))     [Bera-Chakrabarti, prediction-free]
gives a separation ratio  sqrt(#T)  that is LARGE.

But we must keep alpha = oracle-width small WHILE #T grows. In the friendship
graph, #T = s and every triangle shares the hub, so hub has triangle-incidence
degree = 2s -> if we orient toward leaves, hub is NOT a bottleneck (leaves carry
width 1 each). So alpha stays O(1) even as s -> infty. Let's verify:
  - signal = friendship F_s, #T = s (grows)
  - decoy  = K_{d,d}, triangle-free, sets degeneracy d
Choose d = Theta(sqrt(s)) so that m = 3s + d^2 = Theta(s), and #T = s = Theta(m).
Then:
  oracle cost ~ m*alpha/#T = Theta(1)          (tiny!)
  lower bound ~ m/sqrt(#T) = Theta(sqrt(m))     (grows)
Separation ratio ~ sqrt(m). THIS is the honest, large separation.

Also double check alpha really stays O(1) as s grows (the hub concern).
"""
import numpy as np
import networkx as nx

def build(s, d):
    G=nx.Graph(); hub=0; node=1
    for _ in range(s):
        a,b=node,node+1
        G.add_edge(hub,a); G.add_edge(hub,b); G.add_edge(a,b); node+=2
    L=list(range(node,node+d)); R=list(range(node+d,node+2*d))
    for u in L:
        for v in R: G.add_edge(u,v)
    return G

def degeneracy(G):
    H=G.copy(); w=0
    while H.number_of_nodes():
        v=min(H.nodes(),key=lambda x:H.degree(x)); w=max(w,H.degree(v)); H.remove_node(v)
    return w

def oracle_width_K3(G):
    adj={v:set(G.neighbors(v)) for v in G}
    te={frozenset((u,v)):len(adj[u]&adj[v]) for (u,v) in G.edges()}
    tdeg={v:0 for v in G}
    for e,c in te.items():
        u,v=tuple(e); tdeg[u]+=c; tdeg[v]+=c
    ow={v:0 for v in G}
    for (u,v) in G.edges():
        if te[frozenset((u,v))]==0: continue
        ow[u if tdeg[u]<=tdeg[v] else v]+=1
    # report both the max and WHERE it is (hub vs leaf)
    return max(ow.values()), ow[0]  # ow[0] = hub's oriented out-degree

print(f"{'s':>6}{'d':>5}{'m':>8}{'#T':>7}{'kappa':>7}{'alpha':>7}{'hub_out':>8}"
      f"{'oracle~m/T':>11}{'LB~m/sqrtT':>11}{'ratio':>8}")
rows=[]
for s in [200,400,800,1600,3200]:
    d=int(round(s**0.5))
    G=build(s,d); m=G.number_of_edges()
    T=sum(nx.triangles(G).values())//3
    kap=degeneracy(G); a,hub_out=oracle_width_K3(G)
    oc=m*a/T; lb=m/np.sqrt(T); ratio=lb/oc
    rows.append((m,T,kap,a,oc,lb))
    print(f"{s:>6}{d:>5}{m:>8}{T:>7}{kap:>7}{a:>7}{hub_out:>8}"
          f"{oc:>11.2f}{lb:>11.2f}{ratio:>8.2f}")

ms=np.array([r[0] for r in rows])
print(f"\nalpha stays: {[r[3] for r in rows]}  (want O(1) as s grows)")
print(f"oracle cost ~ m^{np.polyfit(np.log(ms),np.log([r[4] for r in rows]),1)[0]:.3f} (want ~0: constant)")
print(f"lower bound ~ m^{np.polyfit(np.log(ms),np.log([r[5] for r in rows]),1)[0]:.3f} (want ~0.5)")
print(f"separation ratio ~ m^{np.polyfit(np.log(ms),np.log([r[5]/r[4] for r in rows]),1)[0]:.3f} (want ~0.5)")
print("\n=> Honest claim: on this family, PredCount w/ perfect predictor uses")
print("   O~(1) space per estimate-unit while any prediction-free O(1)-pass algo")
print("   needs Omega(m/sqrt(#T)) = Omega(sqrt(m)); a polynomial separation.")
