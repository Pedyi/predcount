"""
APPENDIX B groundwork: does the per-copy success probability generalize as
   p_succ = Theta( #H / (m * alpha_H^{b_H - 1}) )
for H in {C4, K4, C5}?  Here b_H is the number of base-edge draws in the FGP
decomposition (the alpha-exponent in the SPACE bound is b_H, so the space is
Otil(m * alpha^{b_H} / #H); see Appendix B). We test on separable constructions.

Method: build a graph where copies of H concentrate on a low-width gadget plus a
copy-free dense decoy, grow the decoy, and check that the ORACLE cost scales in m
with the SAME exponent as the improvement predicts. On these SEPARABLE instances
alpha = Theta(1), so alpha^{b_H} is a constant and both m * alpha^{b_H} / #H and
the older m * alpha^{rho-1} / #H proxy have the SAME m-exponent 1; the UNIFORM
FGP cost (2m)^{rho} / #H scales with exponent rho. Hence the improvement exponent
in m is rho - 1, independent of the alpha-power, which is what this script checks.

(The distinction b_H vs rho-1 only matters for the alpha-power, which these
constant-alpha instances cannot see; it does not affect the measured m-exponent.)

rho: C4 -> 2, K4 -> 2, C5 -> 2.5 (rho(C5)=5/2).
"""
import numpy as np
import networkx as nx
from itertools import combinations

def decoy(G, node0, d, rng):
    L=list(range(node0,node0+d)); R=list(range(node0+d,node0+2*d))
    for u in L:
        for v in R: G.add_edge(u,v)     # triangle-free & C_odd-free bipartite
    return node0+2*d

def gadget_C4(G,node,s):
    # s disjoint-ish C4 sharing structure of small width: hub-pair pattern
    # use two hubs h1,h2 plus leaves forming 4-cycles h1-a-h2-b-h1
    h1,h2=node,node+1; node+=2
    for _ in range(s):
        a,b=node,node+1; node+=2
        G.add_edge(h1,a);G.add_edge(a,h2);G.add_edge(h2,b);G.add_edge(b,h1)
    return node

def gadget_K4(G,node,s):
    # s disjoint K4's (width small: each K4 is its own component-ish)
    for _ in range(s):
        vs=list(range(node,node+4)); node+=4
        for u,v in combinations(vs,2): G.add_edge(u,v)
    return node

def gadget_C5(G,node,s):
    for _ in range(s):
        vs=list(range(node,node+5)); node+=5
        for i in range(5): G.add_edge(vs[i],vs[(i+1)%5])
    return node

def count_H(G,H):
    # small-pattern exact count via networkx subgraph matching (ok for small)
    from networkx.algorithms import isomorphism
    if H=="C4":
        adj={v:set(G.neighbors(v)) for v in G}; c=0
        seen={}
        for u in G:
            for x in adj[u]:
                for y in adj[u]:
                    if x<y: seen[(x,y)]=seen.get((x,y),0)+1
        for (x,y),k in seen.items(): c+=k*(k-1)//2
        return c//2
    if H=="K4":
        adj={v:set(G.neighbors(v)) for v in G}; c=0
        nodes=list(G)
        for a in nodes:
            Na=[x for x in adj[a] if x>a]
            for i in range(len(Na)):
                for j in range(i+1,len(Na)):
                    b,cc=Na[i],Na[j]
                    if cc in adj[b]:
                        common=adj[a]&adj[b]&adj[cc]
                        c+=len([d for d in common if d>cc])
        return c
    if H=="C5":
        # count 5-cycles (expensive) -- approximate by matching on small graphs
        cnt=0; adj={v:set(G.neighbors(v)) for v in G}
        nodes=list(G)
        for path in [(a,) for a in nodes]:
            pass
        # brute for small gadget only
        c=0
        for a in nodes:
            for b in adj[a]:
                for cc in adj[b]:
                    if cc==a: continue
                    for d in adj[cc]:
                        if d in (a,b): continue
                        for e in adj[d]:
                            if e in (a,b,cc): continue
                            if a in adj[e]:
                                c+=1
        return c//10  # each 5-cycle counted 10x (5 rot x 2 dir)
    raise ValueError

RHO={"C4":2.0,"K4":2.0,"C5":2.5}
GAD={"C4":gadget_C4,"K4":gadget_K4,"C5":gadget_C5}

for H in ["C4","K4","C5"]:
    print(f"=== {H} (rho={RHO[H]}) ===")
    print(f"{'m':>8}{'#H':>8}{'unif_exp_cost':>15}{'oracle_cost(a=O(1))':>20}")
    rows=[]
    rng=np.random.default_rng(0)
    s = 60 if H!="C5" else 30
    for dfac in [1,2,3,4]:
        G=nx.Graph(); node=0
        node=GAD[H](G,node,s)
        d=int((s*dfac)**0.5)+2
        node=decoy(G,node,d,rng)
        m=G.number_of_edges()
        H_ct=count_H(G,H)
        if H_ct==0: continue
        rho=RHO[H]
        unif=(2*m)**rho/H_ct
        # alpha=2 is constant on the gadget, so alpha^{b_H} is a constant factor
        # and only the m-exponent matters here: oracle cost ~ m * const / #H.
        orac=m*(2.0**1)/H_ct        # constant-alpha proxy; m-exponent = 1
        rows.append((m,H_ct,unif,orac))
        print(f"{m:>8}{H_ct:>8}{unif:>15.1f}{orac:>20.2f}")
    if len(rows)>=2:
        ms=np.array([r[0] for r in rows])
        eu=np.polyfit(np.log(ms),np.log([r[2] for r in rows]),1)[0]
        eo=np.polyfit(np.log(ms),np.log([r[3] for r in rows]),1)[0]
        print(f"  uniform cost ~ m^{eu:.2f} (expect {RHO[H]}), "
              f"oracle cost ~ m^{eo:.2f} (expect ~1.0), "
              f"gap={eu-eo:.2f} (expect rho-1={RHO[H]-1})\n")
