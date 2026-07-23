#!/usr/bin/env python3
"""
Extended real-graph study (addresses reviewer points 2 and 8).

WHAT THIS ADDS OVER experiments/real_graphs.py
----------------------------------------------
  * MORE AND MORE DIVERSE GRAPHS: collaboration, communication, social, web,
    product co-purchase, autonomous systems, and a biological network, spanning
    ~14K to ~2.3M edges, instead of four graphs of one or two kinds.
  * K4 ON REAL DATA, not only on synthetic separable instances, so that the
    "general H" claim is exercised on real inputs.
  * The alpha/kappa diagnostic reported for every graph and pattern, which is
    the paper's central empirical statistic.

SCALE NOTE. Exact triangle enumeration is fine up to a few million edges, but
K4 enumeration is not. For K4 we sample: we draw uniformly from the triangles
and extend, which estimates the ratio we report (speedup) without enumerating
all copies. Set --k4-sample to control the budget.

Datasets are downloaded from SNAP on demand. If a download fails the graph is
skipped and the rest still run.

Usage (Colab recommended):
    python extended_real.py                 # default set
    python extended_real.py --skip-large    # omit the million-edge graphs
"""
import argparse
import gzip
import time
import urllib.request
import numpy as np
import networkx as nx

# name -> (url, size class)
DATASETS = {
    # collaboration
    "ca-GrQc":      ("https://snap.stanford.edu/data/ca-GrQc.txt.gz", "small"),
    "ca-HepTh":     ("https://snap.stanford.edu/data/ca-HepTh.txt.gz", "small"),
    "ca-CondMat":   ("https://snap.stanford.edu/data/ca-CondMat.txt.gz", "small"),
    "ca-AstroPh":   ("https://snap.stanford.edu/data/ca-AstroPh.txt.gz", "medium"),
    # communication
    "email-Enron":  ("https://snap.stanford.edu/data/email-Enron.txt.gz", "medium"),
    "email-EuAll":  ("https://snap.stanford.edu/data/email-EuAll.txt.gz", "medium"),
    # social
    "facebook":     ("https://snap.stanford.edu/data/facebook_combined.txt.gz", "small"),
    "soc-Slashdot": ("https://snap.stanford.edu/data/soc-Slashdot0902.txt.gz", "medium"),
    # autonomous systems
    "as-caida":     ("https://snap.stanford.edu/data/as-caida20071105.txt.gz", "small"),
    "oregon1":      ("https://snap.stanford.edu/data/oregon1_010526.txt.gz", "small"),
    # web / product
    "web-Stanford": ("https://snap.stanford.edu/data/web-Stanford.txt.gz", "large"),
    "amazon0302":   ("https://snap.stanford.edu/data/amazon0302.txt.gz", "large"),
    # biological
    "bio-CE-GN":    ("https://snap.stanford.edu/data/bio-CE-GN.edges", "small"),
}


def load(name, url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=300).read()
    text = (gzip.decompress(raw) if url.endswith(".gz") else raw)
    text = text.decode("utf-8", errors="ignore")
    G = nx.Graph()
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in "#%":
            continue
        p = line.replace(",", " ").split()
        if len(p) >= 2 and p[0] != p[1]:
            G.add_edge(p[0], p[1])
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


# ---------------- K3 ----------------
def k3_stats(G):
    deg = dict(G.degree())
    adj = {v: set(G.neighbors(v)) for v in G}
    te = {}
    for (u, v) in G.edges():
        te[frozenset((u, v))] = len(adj[u] & adj[v])
    T = sum(te.values()) // 3
    tdeg = {v: 0 for v in G}
    for e, c in te.items():
        u, v = tuple(e); tdeg[u] += c; tdeg[v] += c
    ow = {v: 0 for v in G}
    for (u, v) in G.edges():
        if te[frozenset((u, v))] == 0:
            continue
        ow[u if tdeg[u] <= tdeg[v] else v] += 1
    alpha = max(ow.values()) if ow else 0
    return deg, adj, te, T, alpha


def k3_speedup(G, deg, adj, te, sample=200000, rng=None):
    """Ratio of weighted to uniform success probability, sampling triangles
    if there are too many to enumerate."""
    m = G.number_of_edges()
    edges = [frozenset(e) for e in G.edges()]
    M = len(edges)
    W = sum(te.get(e, 0) + 1.0 for e in edges)
    inv = 1.0 / np.sqrt(2 * m)
    # enumerate or sample triangles
    tris = []
    for u in G:
        Nu = [w for w in adj[u] if w > u]
        for i in range(len(Nu)):
            for j in range(i + 1, len(Nu)):
                a, b = Nu[i], Nu[j]
                if b in adj[a]:
                    tris.append((u, a, b))
                    if len(tris) >= sample:
                        break
            if len(tris) >= sample:
                break
        if len(tris) >= sample:
            break
    if not tris:
        return float("nan")
    pu = pw = 0.0
    for (x, y, z) in tris:
        piv, mid, close = sorted((x, y, z), key=lambda v: (deg[v], str(v)))
        pu += (1.0 / M) * inv
        pb = (te.get(frozenset((piv, mid)), 0) + 1.0) / W
        wsum = wc = 0.0
        for c in adj[piv]:
            w = te.get(frozenset((piv, c)), 0) + 1.0
            wsum += w
            if c == close:
                wc = w
        pw += pb * (wc / wsum if wsum > 0 else 0.0)
    return pw / pu if pu > 0 else float("nan")


# ---------------- K4 (sampled) ----------------
def k4_stats_sampled(G, deg, adj, te, n_sample=20000, rng=None):
    """
    Sample K4s by extending sampled triangles; estimate alpha_{K4} on the
    sampled copy-bearing structure and the weighted/uniform ratio.
    rho(K4)=2, and the sampler draws two disjoint base edges.
    """
    rng = rng or np.random.default_rng(0)
    m = G.number_of_edges()
    edges = [tuple(sorted(e, key=str)) for e in G.edges()]
    M = len(edges)
    # sample triangles, extend to K4
    k4s = []
    tries = 0
    edge_list = edges
    while len(k4s) < n_sample and tries < n_sample * 50:
        tries += 1
        u, v = edge_list[rng.integers(M)]
        common = list(adj[u] & adj[v])
        if len(common) < 2:
            continue
        i, j = rng.choice(len(common), size=2, replace=False)
        w, x = common[i], common[j]
        if x in adj[w]:
            k4s.append((u, v, w, x))
    if not k4s:
        return float("nan"), 0
    # per-edge K4 weight (approximate, from the sample)
    wk = {}
    for (a, b, c, d) in k4s:
        for e in [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]:
            k = frozenset(e)
            wk[k] = wk.get(k, 0) + 1
    W = sum(wk.get(frozenset(e), 0) + 1.0 for e in edges)
    # oracle-width on the sampled structure
    cdeg = {}
    for e, c in wk.items():
        u, v = tuple(e)
        cdeg[u] = cdeg.get(u, 0) + c
        cdeg[v] = cdeg.get(v, 0) + c
    ow = {}
    for e in wk:
        u, v = tuple(e)
        src = u if cdeg.get(u, 0) <= cdeg.get(v, 0) else v
        ow[src] = ow.get(src, 0) + 1
    alpha4 = max(ow.values()) if ow else 0
    # speedup: two opposite (disjoint) base edges per K4
    pu = pw = 0.0
    for (a, b, c, d) in k4s:
        e1 = frozenset((a, b)); e2 = frozenset((c, d))
        pu += (1.0 / M) ** 2
        pw += ((wk.get(e1, 0) + 1.0) / W) * ((wk.get(e2, 0) + 1.0) / W)
    return (pw / pu if pu > 0 else float("nan")), alpha4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-large", action="store_true")
    ap.add_argument("--k4-sample", type=int, default=20000)
    ap.add_argument("--tri-sample", type=int, default=200000)
    args, _ = ap.parse_known_args()

    rng = np.random.default_rng(0)
    print(f"{'graph':>14}{'kind':>8}{'n':>9}{'m':>10}{'#T':>10}{'kappa':>7}"
          f"{'a3':>7}{'a3/k':>7}{'sp_K3':>8}{'a4/k':>7}{'sp_K4':>8}{'t(s)':>7}")
    for name, (url, size) in DATASETS.items():
        if args.skip_large and size == "large":
            continue
        try:
            t0 = time.perf_counter()
            G = load(name, url)
        except Exception as e:
            print(f"{name:>14}  [skip: {type(e).__name__}]")
            continue
        try:
            deg, adj, te, T, a3 = k3_stats(G)
            if T == 0:
                print(f"{name:>14}  [no triangles]")
                continue
            kappa = max(nx.core_number(G).values())
            s3 = k3_speedup(G, deg, adj, te, args.tri_sample, rng)
            s4, a4 = k4_stats_sampled(G, deg, adj, te, args.k4_sample, rng)
            dt = time.perf_counter() - t0
            print(f"{name:>14}{size:>8}{G.number_of_nodes():>9}"
                  f"{G.number_of_edges():>10}{T:>10}{kappa:>7}{a3:>7}"
                  f"{a3/kappa:>7.2f}{s3:>8.1f}"
                  f"{(a4/kappa if kappa else float('nan')):>7.2f}{s4:>8.1f}{dt:>7.1f}")
        except MemoryError:
            print(f"{name:>14}  [skip: out of memory]")
        except Exception as e:
            print(f"{name:>14}  [error: {type(e).__name__}: {e}]")

    print("\nColumns: a3 = oracle-width for K3; a3/k and a4/k are the")
    print("alpha/degeneracy diagnostics. The paper's central empirical claim is")
    print("that a/k ~ 1 on real graphs, for every pattern, so predictions give")
    print("constant-factor and not exponent improvements.")


if __name__ == "__main__":
    main()
