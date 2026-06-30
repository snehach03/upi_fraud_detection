"""
Graph Feature Engineering for UPI Fraud Ring Detection
========================================================
Builds a NetworkX graph from the raw transaction log and computes, for every
account (node), a rich set of structural / behavioural features that are
known to separate fraud-ring participants from genuine users:

  Degree / flow features
    - in_degree, out_degree, in_out_ratio
    - total_in_amount, total_out_amount, net_flow, flow_ratio
    - avg/std transaction amount in & out

  Centrality / topology features
    - pagerank                       (importance / money concentration)
    - betweenness_centrality_approx  (how much a node acts as a "pass-through")
    - clustering_coefficient
    - in_degree / out_degree imbalance -> fan-in/fan-out detector

  Cycle / loop features
    - in_cycle                : node participates in a short directed cycle
    - cycle_length            : length of shortest cycle it's part of

  Community features (Louvain)
    - community_id
    - community_fraud_density : fraction of *known* fraud txns in its community
                                 (computed leave-one-out so it isn't circular)
    - community_size

  Temporal / velocity features
    - txn_burstiness          : txns happening within a short rolling window
    - same_day_in_out_count   : pass-through-same-day pattern typical of mules

The output is a per-account feature table ready to feed a classifier.
"""

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities
from collections import defaultdict


def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for _, row in df.iterrows():
        G.add_edge(row["src"], row["dst"], amount=row["amount"],
                   timestamp=row["timestamp"], txn_id=row["txn_id"])
    return G


def _collapse_to_simple_digraph(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Collapse parallel edges into a single weighted edge (sum of amounts, count)."""
    H = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if H.has_edge(u, v):
            H[u][v]["weight"] += data["amount"]
            H[u][v]["count"] += 1
        else:
            H.add_edge(u, v, weight=data["amount"], count=1)
    return H


def detect_short_cycles(H: nx.DiGraph, max_len=6, cap_cycles=20000):
    """
    Find directed cycles up to length `max_len` (circular money-laundering loops).
    Uses nx.simple_cycles with a length cutoff to stay tractable on larger graphs.
    Returns: dict node -> shortest cycle length it participates in.
    """
    node_cycle_len = {}
    count = 0
    try:
        for cyc in nx.simple_cycles(H, length_bound=max_len):
            count += 1
            if count > cap_cycles:
                break
            L = len(cyc)
            for n in cyc:
                if n not in node_cycle_len or L < node_cycle_len[n]:
                    node_cycle_len[n] = L
    except Exception as e:
        print("Cycle detection warning:", e)
    return node_cycle_len


def compute_features(df: pd.DataFrame, node_labels: dict | None = None) -> pd.DataFrame:
    """
    node_labels: optional dict node -> is_fraud_node (ground truth), used only
    to compute leave-one-out community fraud density for training; not required
    at inference time (defaults to 0 prior).
    """
    G_multi = build_graph(df)
    H = _collapse_to_simple_digraph(G_multi)
    nodes = list(H.nodes())
    print(f"Graph: {H.number_of_nodes()} nodes, {H.number_of_edges()} unique edges "
          f"({G_multi.number_of_edges()} raw transactions)")

    # ---------------- degree / flow ----------------
    in_deg = dict(H.in_degree())
    out_deg = dict(H.out_degree())

    in_amt = defaultdict(list)
    out_amt = defaultdict(list)
    for u, v, d in H.edges(data=True):
        out_amt[u].append(d["weight"])
        in_amt[v].append(d["weight"])

    # ---------------- centrality ----------------
    print("Computing PageRank...")
    pagerank = nx.pagerank(H, alpha=0.85, max_iter=200, weight="weight")

    print("Computing approx betweenness centrality (sampled for speed)...")
    k = min(300, H.number_of_nodes())
    betweenness = nx.betweenness_centrality(H, k=k, normalized=True, seed=42, weight=None)

    print("Computing clustering coefficient...")
    UG = H.to_undirected()
    clustering = nx.clustering(UG)

    # ---------------- cycles ----------------
    print("Detecting short directed cycles (money-loop patterns)...")
    cycle_info = detect_short_cycles(H, max_len=6)

    # ---------------- Louvain communities ----------------
    print("Running Louvain community detection...")
    communities = louvain_communities(UG, weight="weight", seed=42, resolution=1.0)
    node_to_comm = {}
    comm_size = {}
    for i, comm in enumerate(communities):
        comm_size[i] = len(comm)
        for n in comm:
            node_to_comm[n] = i

    # leave-one-out community fraud density (only meaningful if labels supplied)
    comm_fraud_count = defaultdict(int)
    comm_total = defaultdict(int)
    if node_labels:
        for n in nodes:
            c = node_to_comm.get(n, -1)
            comm_total[c] += 1
            if node_labels.get(n, 0) == 1:
                comm_fraud_count[c] += 1

    # ---------------- temporal burstiness ----------------
    print("Computing temporal burstiness features...")
    df_sorted = df.sort_values("timestamp")
    burst_in = defaultdict(int)
    burst_out = defaultdict(int)
    same_day_pass = defaultdict(set)
    for _, r in df_sorted.iterrows():
        same_day_pass[r["dst"]].add((r["timestamp"].date(), "in"))
        same_day_pass[r["src"]].add((r["timestamp"].date(), "out"))

    # rolling burstiness: count txns within 2h windows per node (approx via groupby)
    df_in = df.groupby("dst")["timestamp"].apply(list).to_dict()
    df_out = df.groupby("src")["timestamp"].apply(list).to_dict()

    def max_burst(ts_list, window_minutes=120):
        if not ts_list or len(ts_list) < 2:
            return 0
        ts_sorted = sorted(ts_list)
        max_count = 1
        left = 0
        for right in range(len(ts_sorted)):
            while (ts_sorted[right] - ts_sorted[left]).total_seconds() > window_minutes * 60:
                left += 1
            max_count = max(max_count, right - left + 1)
        return max_count

    rows = []
    for n in nodes:
        ia = in_amt.get(n, [])
        oa = out_amt.get(n, [])
        c = node_to_comm.get(n, -1)
        if node_labels:
            denom = max(comm_total[c] - 1, 1)
            numer = comm_fraud_count[c] - (1 if node_labels.get(n, 0) == 1 else 0)
            comm_density = numer / denom
        else:
            comm_density = 0.0

        same_day_in = sum(1 for d_, t in same_day_pass[n] if t == "in")
        same_day_out = sum(1 for d_, t in same_day_pass[n] if t == "out")

        total_in = sum(ia)
        total_out = sum(oa)

        rows.append(dict(
            node=n,
            in_degree=in_deg.get(n, 0),
            out_degree=out_deg.get(n, 0),
            degree_ratio=(in_deg.get(n, 0) + 1) / (out_deg.get(n, 0) + 1),
            total_in_amount=total_in,
            total_out_amount=total_out,
            net_flow=total_in - total_out,
            flow_ratio=(total_in + 1) / (total_out + 1),
            avg_in_amount=float(np.mean(ia)) if ia else 0.0,
            avg_out_amount=float(np.mean(oa)) if oa else 0.0,
            std_in_amount=float(np.std(ia)) if ia else 0.0,
            std_out_amount=float(np.std(oa)) if oa else 0.0,
            pagerank=pagerank.get(n, 0.0),
            betweenness=betweenness.get(n, 0.0),
            clustering_coeff=clustering.get(n, 0.0),
            in_cycle=int(n in cycle_info),
            cycle_length=cycle_info.get(n, 0),
            community_id=c,
            community_size=comm_size.get(c, 1),
            community_fraud_density=comm_density,
            burst_in_2h=max_burst(df_in.get(n, [])),
            burst_out_2h=max_burst(df_out.get(n, [])),
            same_day_pass_through=min(same_day_in, same_day_out),
            fan_in_score=in_deg.get(n, 0) / max(out_deg.get(n, 0), 1),
            fan_out_score=out_deg.get(n, 0) / max(in_deg.get(n, 0), 1),
        ))

    feat_df = pd.DataFrame(rows)
    return feat_df, H, node_to_comm


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/upi_fraud_graph/data/transactions.csv", parse_dates=["timestamp"])
    # ground truth node labels (for offline eval / leave-one-out community density)
    node_labels = {}
    for _, r in df.iterrows():
        if r["is_fraud"] == 1:
            node_labels[r["src"]] = 1
            node_labels[r["dst"]] = 1
    feat_df, H, node_to_comm = compute_features(df, node_labels)
    feat_df["is_fraud_node"] = feat_df["node"].map(lambda n: node_labels.get(n, 0))
    feat_df.to_csv("/home/claude/upi_fraud_graph/data/node_features.csv", index=False)
    print(feat_df.shape)
    print(feat_df["is_fraud_node"].value_counts())
