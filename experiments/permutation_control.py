"""How much of the learned predictor's gain is real STRUCTURE vs just the
marginal weight distribution? Compare learned vs permuted-weight control."""
import numpy as np, networkx as nx, validate_learned as vl
rng=np.random.default_rng(0)
graphs = vl.synthetic_graphs()
names=list(graphs)
cols=vl.FEATURE_GROUPS["both"]
rows=[]
for tr in names:
    e,X = vl.features(graphs[tr]); y=vl.targets(graphs[tr],e)
    th=vl.ridge(X[:,cols],y,1.0)
    for te in names:
        if te==tr: continue
        r=vl.evaluate_on(graphs[te],th,cols,rng)
        base=1.0
        # what fraction of the ABOVE-BASELINE gain is structural?
        gain_learned = r["learned"]-base
        gain_random  = r["perm"].mean()-base
        struct_share = (gain_learned-gain_random)/gain_learned
        rows.append((r["perfect"],r["learned"],r["perm"].mean(),struct_share))
        print(f"{tr:>10}->{te:<10} perfect={r['perfect']:6.1f} "
              f"learned={r['learned']:6.1f} random={r['perm'].mean():6.1f} "
              f"structural share={struct_share:5.1%}")
ss=np.array([r[3] for r in rows])
print(f"\nStructural share of the learned gain: mean {ss.mean():.1%}, "
      f"range [{ss.min():.1%}, {ss.max():.1%}]")
print("\nIf this share is SMALL, most of the speedup comes from the weight")
print("DISTRIBUTION (heavy-tailed weights concentrate sampling) rather than")
print("from correctly identifying WHICH edges are heavy. That is an important")
print("caveat and must be reported.")
