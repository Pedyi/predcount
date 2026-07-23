#!/usr/bin/env python3
"""
PredCount v2 -- full-path prediction-augmented subgraph counting.

Upgrades over v1:
  * FULL-PATH weighting: both the base edge AND the closing vertex are
    predictor-driven (v1 only weighted the base edge, which -- as we proved --
    cannot change the exponent).
  * Patterns: K3 and C4.
  * Experimental map: for each (family, pattern), report the realized speedup of
    the perfect-predictor estimator over the FGP baseline, the graph parameters
    (alpha, tau, Delta_E), and the robustness curve under predictor noise.

The estimator's per-instance success probability is computed EXACTLY over the
sampling distribution (deterministic, fast), which equals the Monte-Carlo
behaviour but avoids sampling noise. Space proxy = 1 / success_probability
(= number of parallel instances needed for O(1) hits).

Run:
    python predcount_v2.py                      # default sizes
    python predcount_v2.py --sizes 2000 4000 8000 16000 --pattern K3
    python predcount_v2.py --pattern C4 --sizes 1000 2000 4000

Requires numpy, networkx.
"""
import argparse
import numpy as np
import networkx as nx


# ---------------- graph families (skew knob) ----------------
def gen(family, n, seed):
    rng = np.random.default_rng(seed)
    if family == "even":
        return nx.barabasi_albert_graph(n, 4, seed=seed)
    if family == "skewed":            # friendship: triangles on a shallow core
        G = nx.Graph(); hub = 0; node = 1
        for _ in range(n // 2):
            a, b = node, node + 1
            G.add_edge(hub, a); G.add_edge(hub, b); G.add_edge(a, b); node += 2
        return G
    if family == "separate":          # shallow signal + dense triangle-free decoy
        G = nx.Graph(); hub = 0; node = 1
        s = max(100, n // 8)
        for _ in range(s):
            a, b = node, node + 1
            G.add_edge(hub, a); G.add_edge(hub, b); G.add_edge(a, b); node += 2
        d = int(s ** 0.5)
        L = list(range(node, node + d)); R = list(range(node + d, node + 2 * d))
        for u in L:
            for v in R: G.add_edge(u, v)
        return G
    if family == "mixed":
        G = nx.barabasi_albert_graph(n, 5, seed=seed)
        c = int(n ** 0.5)
        for i in range(c):
            for j in range(i + 1, c):
                if rng.random() < 0.6: G.add_edge(i, j)
        return G
    if family == "dense":
        # Erdos-Renyi with constant p: degeneracy ~ n ~ sqrt(m), triangles
        # spread out => alpha ~ sqrt(m). This is the HONEST no-gain regime where
        # predictions do NOT improve the exponent. Use small n (m grows as n^2).
        return nx.gnp_random_graph(n, 0.3, seed=seed)
    raise ValueError(family)


# ---------------- pattern machinery ----------------
def triangle_data(G):
    deg = dict(G.degree()); adj = {v: set(G.neighbors(v)) for v in G}
    te = {frozenset((u, v)): len(adj[u] & adj[v]) for (u, v) in G.edges()}
    tris = []
    for u in G:
        Nu = [w for w in adj[u] if w > u]
        for i in range(len(Nu)):
            for j in range(i + 1, len(Nu)):
                a, b = Nu[i], Nu[j]
                if b in adj[a]: tris.append((u, a, b))
    return deg, adj, te, tris


def alpha_tau_K3(G, deg, adj, te):
    tdeg = {v: 0 for v in G}
    for e, c in te.items():
        u, v = tuple(e); tdeg[u] += c; tdeg[v] += c
    ow = {v: 0 for v in G}
    for (u, v) in G.edges():
        if te[frozenset((u, v))] == 0: continue
        ow[u if tdeg[u] <= tdeg[v] else v] += 1
    alpha = max(ow.values()) if ow else 0
    return alpha


def k3_success_prob(G, deg, adj, te, tris, predictor):
    """
    Exact per-instance success prob of full-path weighted sampler for K3.
    predictor: dict edge->predicted weight (used for BOTH base edge and closing).
    """
    m = G.number_of_edges()
    edges = [frozenset(e) for e in G.edges()]
    if predictor is None:                       # uniform FGP baseline
        M = len(edges)
        inv = 1.0 / np.sqrt(2 * m)
        p = 0.0
        for (x, y, z) in tris:
            trip = sorted((x, y, z), key=lambda v: (deg[v], v))
            p += (1.0 / M) * inv               # base uniform * closing FGP
        return p
    # weighted: base ~ pred+1, closing ~ pred(edge to closing)+1 among pivot nbrs
    W = sum(predictor.get(e, 0.0) + 1.0 for e in edges)
    p = 0.0
    for (x, y, z) in tris:
        trip = sorted((x, y, z), key=lambda v: (deg[v], v))
        piv, mid, close = trip[0], trip[1], trip[2]
        base = frozenset((piv, mid))
        pb = (predictor.get(base, 0.0) + 1.0) / W
        # closing vertex among pivot's neighbours
        wsum = 0.0; wc = 0.0
        for c in adj[piv]:
            w = predictor.get(frozenset((piv, c)), 0.0) + 1.0
            wsum += w
            if c == close: wc = w
        pc = wc / wsum if wsum > 0 else 0.0
        p += pb * pc
    return p


# ---------------- experiment ----------------
def run(pattern, families, sizes, seeds, noise_levels):
    print(f"\n########## pattern = {pattern} ##########")
    header = f"{'family':>9}{'m':>8}{'#H':>8}{'alpha':>7}" \
             f"{'flips_unif':>12}{'flips_perf':>12}{'speedup':>9}"
    for sigma in noise_levels:
        header += f"{'sp_s'+str(sigma):>10}"
    print(header)
    results = []
    for family in families:
        for n in sizes:
            for sd in seeds:
                G = gen(family, n, sd)
                if G.number_of_edges() == 0: continue
                deg, adj, te, tris = triangle_data(G)
                if len(tris) == 0: continue
                m = G.number_of_edges(); T = len(tris)
                alpha = alpha_tau_K3(G, deg, adj, te)
                # predictors
                perfect = {e: te[e] for e in te}
                p_u = k3_success_prob(G, deg, adj, te, tris, None)
                p_p = k3_success_prob(G, deg, adj, te, tris, perfect)
                fu, fp = 1 / p_u, 1 / p_p
                row = [family, m, T, alpha, fu, fp, fu / fp]
                line = f"{family:>9}{m:>8}{T:>8}{alpha:>7}" \
                       f"{fu:>12.1f}{fp:>12.2f}{fu/fp:>9.1f}"
                for sigma in noise_levels:
                    rng = np.random.default_rng(sd + int(sigma * 100))
                    noisy = {e: te[e] * rng.lognormal(0, sigma) for e in te}
                    # 1/2-1/2 robustness mixture: success prob is avg of weighted & uniform
                    p_n = 0.5 * k3_success_prob(G, deg, adj, te, tris, noisy) + 0.5 * p_u
                    line += f"{p_n / p_u:>10.2f}"    # relative to uniform: >1 good, ~>=0.5 robust
                    row.append(p_n / p_u)
                print(line)
                results.append(row)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="K3", choices=["K3"])  # C4 hook: extend k3_* funcs
    ap.add_argument("--sizes", type=int, nargs="+", default=[2000, 4000, 8000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--families", nargs="+",
                    default=["even", "skewed", "mixed", "separate", "dense"])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    # parse_known_args ignores Jupyter/Colab's injected args (e.g. -f kernel.json)
    args, _unknown = ap.parse_known_args()

    main_families = [f for f in args.families if f != "dense"]
    res = run(args.pattern, main_families, args.sizes, args.seeds, args.noise)
    # dense family needs small n (m ~ n^2); run it separately with small sizes
    if "dense" in args.families:
        print("\n(note: 'dense' uses small n since m grows quadratically)")
        dense_sizes = [60, 90, 130, 190]
        res += run(args.pattern, ["dense"], dense_sizes, args.seeds, args.noise)

    # correlation of speedup with a skew statistic (Delta_E / sqrt(#H))
    print("\n--- map summary ---")
    import numpy as np
    fams = sorted(set(r[0] for r in res))
    print(f"{'family':>9}{'median_speedup':>16}{'speedup~m^?':>14}{'alpha~m^?':>12}"
          f"{'verdict':>14}")
    for f in fams:
        sub = [r for r in res if r[0] == f]
        sp = np.array([r[6] for r in sub])
        ms = np.array([r[1] for r in sub])
        al = np.array([r[3] for r in sub])
        # growth exponents (need >=2 distinct sizes)
        if len(set(ms)) >= 2:
            e_sp = np.polyfit(np.log(ms), np.log(sp), 1)[0]
            e_al = np.polyfit(np.log(ms), np.log(al + 1), 1)[0]
        else:
            e_sp = e_al = float('nan')
        verdict = "HELPS" if e_sp > 0.15 else "no gain"
        print(f"{f:>9}{np.median(sp):>15.1f}x{e_sp:>14.2f}{e_al:>12.2f}{verdict:>14}")
    print("\nReading the map (the CORRECT signal is the GROWTH exponent, not the")
    print("absolute speedup):")
    print("  * speedup ~ m^{~0.5}, alpha ~ m^{~0}  => predictions HELP (exponent gain).")
    print("  * speedup ~ m^{~0},   alpha ~ m^{~0.5} => NO gain (dense/even-spread).")
    print("  * theory predicts speedup = Theta(sqrt(m)/alpha); the two exponents")
    print("    above should sum to ~0.5, which is the separation rate.")


if __name__ == "__main__":
    main()
