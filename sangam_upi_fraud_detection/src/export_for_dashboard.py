"""
Export everything the dashboard needs as static JSON: node features + predictions,
edges (for graph viz), detected rings, and metrics. This lets the dashboard be a
pure static HTML/JS app (zero server dependency -> trivially deployable anywhere:
GitHub Pages, Netlify, Vercel, S3, or just opening the file).
"""
import json
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from graph_features import compute_features
from fraud_classifier import FEATURE_COLS, rule_based_ring_flags

def main():
    df = pd.read_csv("/home/claude/upi_fraud_graph/data/transactions.csv", parse_dates=["timestamp"])
    node_labels = {}
    for _, r in df.iterrows():
        if r["is_fraud"] == 1:
            node_labels[r["src"]] = 1
            node_labels[r["dst"]] = 1

    feat_df, H, node_to_comm = compute_features(df, node_labels)
    feat_df["is_fraud_node"] = feat_df["node"].map(lambda n: node_labels.get(n, 0))

    artifacts = joblib.load("/home/claude/upi_fraud_graph/models/fraud_model.joblib")
    model, scaler = artifacts["model"], artifacts["scaler"]
    X = feat_df[FEATURE_COLS].fillna(0)
    Xs = scaler.transform(X)
    feat_df["predicted_fraud"] = model.predict(Xs)
    feat_df["fraud_probability"] = model.predict_proba(Xs)[:, 1]

    feat_df = rule_based_ring_flags(feat_df)

    # account_type for display
    acc_type = {}
    for n in H.nodes():
        sub = df[(df["src"] == n) | (df["dst"] == n)]
        if len(sub):
            pat = sub["pattern"].mode()
            acc_type[n] = pat.iloc[0] if len(pat) else "unknown"
        else:
            acc_type[n] = "unknown"
    feat_df["sample_pattern"] = feat_df["node"].map(acc_type)

    # detected rings: connected components among ML-flagged accounts
    flagged = feat_df[feat_df["predicted_fraud"] == 1]["node"].tolist()
    sub_g = H.subgraph(flagged).to_undirected()
    rings = []
    for i, comp in enumerate(nx.connected_components(sub_g)):
        if len(comp) >= 2:
            members = sorted(comp)
            ring_amount = 0.0
            for u, v, d in H.subgraph(comp).edges(data=True):
                ring_amount += d.get("weight", 0)
            rings.append(dict(
                ring_id=f"RING_{i:03d}",
                size=len(members),
                members=members[:50],
                total_flow=round(ring_amount, 2),
                avg_fraud_prob=round(float(feat_df[feat_df["node"].isin(members)]["fraud_probability"].mean()), 4)
            ))
    rings.sort(key=lambda r: -r["size"])

    # downsample edges for visualization (cap for browser performance)
    viz_df = df.copy()
    # always include all fraud edges + sample of legit edges
    fraud_edges = viz_df[viz_df["is_fraud"] == 1]
    legit_sample = viz_df[viz_df["is_fraud"] == 0].sample(n=min(800, (viz_df["is_fraud"]==0).sum()), random_state=42)
    viz_edges = pd.concat([fraud_edges, legit_sample])

    edges_out = [
        dict(src=r["src"], dst=r["dst"], amount=round(float(r["amount"]), 2),
             is_fraud=int(r["is_fraud"]), pattern=r["pattern"], ts=str(r["timestamp"]))
        for _, r in viz_edges.iterrows()
    ]

    nodes_in_viz = set(viz_edges["src"]) | set(viz_edges["dst"])
    nodes_out = []
    for _, r in feat_df[feat_df["node"].isin(nodes_in_viz)].iterrows():
        nodes_out.append(dict(
            id=r["node"], in_degree=int(r["in_degree"]), out_degree=int(r["out_degree"]),
            pagerank=round(float(r["pagerank"]), 6), community=int(r["community_id"]),
            predicted_fraud=int(r["predicted_fraud"]), fraud_probability=round(float(r["fraud_probability"]), 4),
            is_fraud_node=int(r["is_fraud_node"]), pattern=r["sample_pattern"],
            in_cycle=int(r["in_cycle"])
        ))

    metrics = json.load(open("/home/claude/upi_fraud_graph/models/metrics.json"))

    summary_stats = dict(
        total_accounts=int(len(feat_df)),
        total_transactions=int(len(df)),
        total_fraud_accounts_true=int(feat_df["is_fraud_node"].sum()),
        total_flagged_by_model=int(feat_df["predicted_fraud"].sum()),
        rings_detected=len(rings),
        total_amount_in_flagged_rings=round(sum(r["total_flow"] for r in rings), 2),
        metrics=metrics,
    )

    output = dict(
        summary=summary_stats,
        nodes=nodes_out,
        edges=edges_out,
        rings=rings[:30],
        feature_importance=artifacts["importances"],
    )

    with open("/home/claude/upi_fraud_graph/app/dashboard_data.json", "w") as f:
        json.dump(output, f)

    feat_df.to_csv("/home/claude/upi_fraud_graph/data/node_features_scored.csv", index=False)
    print(f"Exported {len(nodes_out)} nodes, {len(edges_out)} edges, {len(rings)} rings to dashboard_data.json")
    print(json.dumps(summary_stats, indent=2))

if __name__ == "__main__":
    main()
