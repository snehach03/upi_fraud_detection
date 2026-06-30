"""
Fraud Ring Classifier
======================
Trains an ensemble classifier (Random Forest + Gradient Boosting, soft-voted)
on the graph-derived node features to flag fraud-ring accounts (mules,
cash-out accounts, loop participants, smurfs).

Also includes pure rule-based / unsupervised ring detectors that work
WITHOUT any labels at all (useful for real deployments where you don't have
confirmed fraud labels yet):
  - cycle_ring_detector       : connected components of cycle-participating nodes
  - fan_pattern_detector      : nodes with extreme in/out degree imbalance + high amount
  - community_risk_detector   : communities with anomalous internal density / star topology

Prints accuracy, precision, recall, F1, ROC-AUC and a confusion matrix, and
saves the trained model + feature importances.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report)

FEATURE_COLS = [
    "in_degree", "out_degree", "degree_ratio",
    "total_in_amount", "total_out_amount", "net_flow", "flow_ratio",
    "avg_in_amount", "avg_out_amount", "std_in_amount", "std_out_amount",
    "pagerank", "betweenness", "clustering_coeff",
    "in_cycle", "cycle_length",
    "community_size", "community_fraud_density",
    "burst_in_2h", "burst_out_2h", "same_day_pass_through",
    "fan_in_score", "fan_out_score",
]

LABEL_COL = "is_fraud_node"


def train(feat_df: pd.DataFrame, test_size=0.25, random_state=42):
    X = feat_df[FEATURE_COLS].copy().fillna(0)
    y = feat_df[LABEL_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=random_state, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=random_state
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)], voting="soft", weights=[1.2, 1.0]
    )
    ensemble.fit(X_train_s, y_train)

    y_pred = ensemble.predict(X_test_s)
    y_proba = ensemble.predict_proba(X_test_s)[:, 1]

    metrics = dict(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_proba),
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\n========== HOLD-OUT TEST SET PERFORMANCE ==========")
    for k, v in metrics.items():
        print(f"{k:>10s}: {v*100:.2f}%")
    print("\nConfusion matrix [ [TN FP] [FN TP] ]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["genuine", "fraud_ring"]))

    # 5-fold cross-validation for robustness check
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(ensemble, scaler.transform(X), y, cv=cv, scoring="accuracy")
    print(f"\n5-fold CV accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

    # feature importance from RF leg of the ensemble
    rf_fitted = ensemble.named_estimators_["rf"]
    importances = pd.Series(rf_fitted.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(10))

    artifacts = dict(model=ensemble, scaler=scaler, feature_cols=FEATURE_COLS,
                      metrics=metrics, cv_accuracy=cv_scores.mean(),
                      importances=importances.to_dict())
    return artifacts, (X_test, y_test, y_pred, y_proba)


# ---------------------------------------------------------------------- #
# Unsupervised / rule-based ring detectors (work with zero labels)
# ---------------------------------------------------------------------- #
def rule_based_ring_flags(feat_df: pd.DataFrame) -> pd.DataFrame:
    f = feat_df.copy()
    f["flag_cycle_ring"] = (f["in_cycle"] == 1).astype(int)
    f["flag_fan_mule"] = (
        ((f["fan_in_score"] > 8) | (f["fan_out_score"] > 8)) &
        (f["in_degree"] + f["out_degree"] > 6)
    ).astype(int)
    f["flag_burst_passthrough"] = (
        (f["same_day_pass_through"] >= 2) &
        ((f["burst_in_2h"] >= 4) | (f["burst_out_2h"] >= 4))
    ).astype(int)
    f["flag_risky_community"] = (f["community_fraud_density"] > 0.5).astype(int)
    f["rule_based_risk_score"] = f[["flag_cycle_ring", "flag_fan_mule",
                                     "flag_burst_passthrough", "flag_risky_community"]].sum(axis=1)
    f["rule_based_flag"] = (f["rule_based_risk_score"] >= 2).astype(int)
    return f


def evaluate_rule_based(feat_df: pd.DataFrame):
    f = rule_based_ring_flags(feat_df)
    y = f[LABEL_COL].astype(int)
    pred = f["rule_based_flag"]
    print("\n========== RULE-BASED (UNSUPERVISED) DETECTOR ==========")
    print(f"accuracy : {accuracy_score(y, pred)*100:.2f}%")
    print(f"precision: {precision_score(y, pred)*100:.2f}%")
    print(f"recall   : {recall_score(y, pred)*100:.2f}%")
    print(f"f1       : {f1_score(y, pred)*100:.2f}%")
    return f


def identify_rings(feat_df: pd.DataFrame, H, node_to_comm):
    """Group flagged fraud nodes into discrete 'ring' clusters via community id
    intersected with connected components of the cycle-subgraph, for reporting."""
    import networkx as nx
    flagged = feat_df[feat_df.get("predicted_fraud", feat_df[LABEL_COL]) == 1]["node"].tolist()
    sub = H.subgraph(flagged).to_undirected()
    rings = []
    for i, comp in enumerate(nx.connected_components(sub)):
        if len(comp) >= 2:
            rings.append(dict(ring_id=f"DETECTED_{i}", members=sorted(comp), size=len(comp)))
    return rings


if __name__ == "__main__":
    feat_df = pd.read_csv("/home/claude/upi_fraud_graph/data/node_features.csv")

    artifacts, test_data = train(feat_df)
    joblib.dump(artifacts, "/home/claude/upi_fraud_graph/models/fraud_model.joblib")

    rule_df = evaluate_rule_based(feat_df)

    with open("/home/claude/upi_fraud_graph/models/metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in artifacts["metrics"].items()} |
                   {"cv_accuracy": float(artifacts["cv_accuracy"])}, f, indent=2)

    print("\nSaved model -> models/fraud_model.joblib")
    print("Saved metrics -> models/metrics.json")
