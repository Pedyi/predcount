"""
CONSISTENCY MECHANISM CHECK (per-copy, full path) -- for K3.

Goal: verify the heart of Theorem (consistency). Under a PERFECT predictor that
weights BOTH sampling steps (base edge AND closing vertex), the per-instance
success probability of finding SOME triangle should scale like
      p_success  ~  #T / (m * alpha)     for K3 (b_{K3} = 1 base-edge draw),
i.e. flips-to-success ~ (m*alpha)/#T, versus the uniform baseline (2m)^{1.5}/#T.
(This is Lemma 9 / Theorem 10: the alpha-exponent is the number of base-edge
draws b_H, which is 1 for K3 -- NOT rho-1=0.5.) On a separable family alpha is
O(1), so flips-to-success ~ m and the IMPROVEMENT exponent in m is rho-1 = 0.5.

We compute p_success EXACTLY (no Monte Carlo) under three samplers:
  (U)  uniform base edge (1/m), uniform-FGP closing vertex (1/sqrt(2m))
  (W1) weighted base edge (~ true edge triangle-count), FGP closing vertex
  (W2) weighted base edge AND weighted closing vertex (both predictor-driven)
and report flips-to-success = 1/p_success and its exponent in m on the
'separate' family (alpha=O(1)) and on an 'even' family (alpha large).

If W2's exponent in m drops to ~1.0 on the separate family while U stays ~1.5,
the per-copy mechanism behind consistency is confirmed.

Requires: numpy, networkx.
"""
import numpy as np
import networkx as nx

def build_separate(s, seed=0):
    d=int(round(s**0.5))
    G=nx.Graph(); hub=0; node=1
    for _ in range(s):
        a,b=node,node+1
        G.add_edge(hub,a); G.add_edge(hub,b); G.add_edge(a,b); node+=2
    L=list(range(node,node+d)); R=list(range(node+d,node+2*d))
    for u in L:
        for v in R: G.add_edge(u,v)
    return G

def build_even(n, seed=0):
    return nx.barabasi_albert_graph(n, 4, seed=seed)

def analyze(G):
    m=G.number_of_edges()
    deg=dict(G.degree())
    adj={v:set(G.neighbors(v)) for v in G}
    edges=[tuple(sorted(e)) for e in G.edges()]
    eidx={e:i for i,e in enumerate(edges)}
    M=len(edges)
    # true triangles-per-edge (predictor target)
    te=np.array([len(adj[e[0]]&adj[e[1]]) for e in edges], dtype=float)
    # enumerate triangles + canonical route (base edge=min-deg pair, closing=3rd)
    tris=[]
    for u in G:
        Nu=[w for w in adj[u] if w>u]
        for i in range(len(Nu)):
            for j in range(i+1,len(Nu)):
                a,b=Nu[i],Nu[j]
                if b in adj[a]: tris.append((u,a,b))
    T=len(tris)
    if T==0: return None

    def psucc(mode):
        # sum over triangles of Pr[this canonical route is realized]
        # base edge prob:
        if mode=="U":
            qb=np.full(M,1.0/M)
        else:
            w=te+1.0; qb=w/w.sum()
        # closing vertex prob per route:
        #   FGP-uniform: 1/sqrt(2m)
        #   weighted   : proportional to (te of the closing edge)+1 among neighbors
        p=0.0
        inv_sqrt=1.0/np.sqrt(2*m)
        for (x,y,z) in tris:
            # canonical: base=(min-deg pair), low endpoint, closing = remaining
            trip=sorted((x,y,z), key=lambda v:(deg[v],v))
            low=trip[0]; mid=trip[1]; close=trip[2]
            base=tuple(sorted((low,mid)))
            kb=eidx[base]
            pb=qb[kb]
            if mode=="W2":
                # weight the closing vertex among low's neighbors by edge-te+1
                # (predictor knows which neighbor closes a heavy triangle)
                nbr=list(adj[low])
                wsum=0.0; wclose=0.0
                for c in nbr:
                    ec=frozenset((low,c))
                    wc=(len(adj[low]&adj[c])+1.0)
                    wsum+=wc
                    if c==close: wclose=wc
                pc=wclose/wsum if wsum>0 else 0.0
            else:
                pc=inv_sqrt
            p+=pb*pc
        return p

    out={}
    for mode in ["U","W1","W2"]:
        ps=psucc(mode)
        out[mode]=(ps, 1.0/ps if ps>0 else float('inf'))
    return m,T,out

print("=== SEPARATE family (alpha=O(1)) ===")
print(f"{'m':>8}{'#T':>7}{'flipsU':>12}{'flipsW1':>12}{'flipsW2':>12}")
sep=[]
for s in [200,400,800,1600,3200]:
    r=analyze(build_separate(s))
    if r is None: continue
    m,T,o=r; sep.append((m,o))
    print(f"{m:>8}{T:>7}{o['U'][1]:>12.1f}{o['W1'][1]:>12.1f}{o['W2'][1]:>12.1f}")
ms=np.array([r[0] for r in sep])
for mode in ["U","W1","W2"]:
    e=np.polyfit(np.log(ms),np.log([r[1][mode][1] for r in sep]),1)[0]
    print(f"  flips[{mode}] ~ m^{e:.3f}")

print("\n=== EVEN family (alpha large) ===")
print(f"{'m':>8}{'#T':>7}{'flipsU':>12}{'flipsW1':>12}{'flipsW2':>12}")
ev=[]
for n in [1000,2000,4000,8000]:
    r=analyze(build_even(n))
    if r is None: continue
    m,T,o=r; ev.append((m,o))
    print(f"{m:>8}{T:>7}{o['U'][1]:>12.1f}{o['W1'][1]:>12.1f}{o['W2'][1]:>12.1f}")
ms=np.array([r[0] for r in ev])
for mode in ["U","W1","W2"]:
    e=np.polyfit(np.log(ms),np.log([r[1][mode][1] for r in ev]),1)[0]
    print(f"  flips[{mode}] ~ m^{e:.3f}")

print("\nExpectation: on SEPARATE, flips[W2] exponent ~1.0 (vs U ~1.5) => the")
print("full-path weighting achieves the consistency success prob. On EVEN, all")
print("three stay close (no separable structure) => predictions don't help,")
print("matching the honest scope of the theorem.")
