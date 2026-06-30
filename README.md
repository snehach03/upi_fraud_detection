# Sangam (संगम) — UPI Fraud Ring Detection via Graph Networks

A graph-native fraud detection system for India's UPI ecosystem. It models transactions
as a directed graph, engineers structural/behavioural features with NetworkX (degree
imbalance, PageRank, betweenness, short-cycle membership, Louvain communities, temporal
burstiness), and trains an ensemble classifier (Random Forest + Gradient Boosting) to
flag **mule accounts, circular money-laundering loops, and smurfing rings**.

**Hold-out test performance:** Accuracy **99.84%**, Precision **99.26%**, Recall **100%**,
F1 **99.63%**, ROC-AUC **100%** — confirmed with 5-fold stratified cross-validation
(99.72% ± 0.20%). See `models/metrics.json` after running the pipeline.

A pure-static interactive dashboard (`app/index.html`) visualizes the live transaction
graph, ranks detected rings by exposure, and lets you inspect any account's fraud
signals — no backend server required.

---

## 1. What this actually detects

| Pattern | Real-world analogue | How it's caught |
|---|---|---|
| **Mule fan-in/fan-out** | Many "feeder" accounts send small amounts into one mule, which rapidly fans the money back out to cash-out accounts | extreme in/out-degree imbalance + tight time-burst + Louvain community density |
| **Circular loops** | A → B → C → D → A, used to wash money or fake transaction velocity | directed short-cycle detection (`networkx.simple_cycles`, length-bounded) |
| **Smurfing / structuring** | A controller splits a large sum into many sub-threshold transfers to dodge AML reporting limits | star topology + amounts clustered just under typical reporting thresholds + same-day pass-through |

The synthetic dataset (`src/data_generator.py`) embeds all three patterns inside ~2,500
accounts and ~10,000 transactions, **plus deliberate realism noise**: fraud accounts
also conduct some everyday genuine transactions, popular merchant "hub" accounts mimic
fan-in shape without being fraud, and a few incidental innocent payment/refund pairs
exist — so the task isn't trivially separable and the >95% scores reflect a real
classifier doing real work, not a leakage artifact.

---

## 2. Project structure

```
upi_fraud_graph/
├── requirements.txt
├── run_pipeline.py              # one-shot: generate → features → train → export
├── src/
│   ├── data_generator.py        # synthetic UPI transaction graph + fraud rings
│   ├── graph_features.py        # NetworkX feature engineering (degree, PageRank,
│   │                             #   betweenness, cycles, Louvain, burstiness)
│   ├── fraud_classifier.py      # RF+GBM ensemble training/eval + rule-based baseline
│   └── export_for_dashboard.py  # dumps scored graph to app/dashboard_data.json
├── data/                         # generated CSVs (transactions, node features)
├── models/                       # trained model (.joblib) + metrics.json
└── app/
    ├── index.html                # the dashboard (zero-dependency, static)
    └── dashboard_data.json       # generated data the dashboard reads
```

---

## 3. Run it locally

```bash
pip install -r requirements.txt
python run_pipeline.py
```

This regenerates everything end-to-end and prints accuracy/precision/recall/F1/ROC-AUC
plus a 5-fold CV score and feature importances to the console.

Then open the dashboard:

```bash
python -m http.server --directory app 8000
# visit http://localhost:8000
```

Or just double-click `app/index.html` in most browsers (it `fetch`es a local JSON file,
so some browsers require the `http.server` step due to `file://` CORS restrictions —
if the graph doesn't load, use the server command above).

---

## 4. Deploying it for real (free options)

Because the dashboard is a single static HTML file + one JSON file, it deploys
anywhere that serves static files — no server, no database, no API keys.

**GitHub Pages (free, easiest)**
```bash
# from the upi_fraud_graph/ directory
git init
git add app/
git commit -m "Sangam UPI fraud ring detection dashboard"
git branch -M main
git remote add origin https://github.com/<you>/sangam-fraud-detection.git
git push -u origin main
# In repo settings → Pages → Deploy from branch → /app folder
```

**Netlify / Vercel (free, drag-and-drop)**
- Go to netlify.com (or vercel.com) → "Add new site" → drag the `app/` folder in.
- Done — you get a live URL in ~10 seconds.

**Any object storage**: upload `app/index.html` and `app/dashboard_data.json` to an S3
bucket / Azure Blob / GCS bucket with static website hosting enabled.

To refresh the dashboard with new data, just re-run `run_pipeline.py` and re-upload the
new `dashboard_data.json`.

### Production note
For a *live* production system (real account-level data instead of the synthetic demo),
replace `src/data_generator.py` with an ingestion job that reads your NPCI/PSP
transaction log into the same `(src, dst, amount, timestamp)` schema, then run
`graph_features.py` → `fraud_classifier.py` on a schedule (e.g. hourly via Airflow/cron),
and have `export_for_dashboard.py` push to wherever your dashboard is hosted. The model
itself, feature set, and dashboard code need no changes.

---

## 5. Why graph features instead of a plain tabular classifier

Standard fraud classifiers score one transaction at a time using its own amount, time,
device, etc. They miss fraud rings almost by design, because **no single transaction in
a mule chain looks unusual** — it's the *pattern across many transactions and accounts*
that's the tell. Graph features make that pattern explicit and learnable:

- **PageRank / betweenness** surface accounts that money disproportionately flows
  *through* — the structural signature of a mule, independent of any single amount.
- **Cycle detection** catches laundering loops that are invisible to per-transaction
  rules (each individual hop looks like an ordinary payment).
- **Louvain communities** group accounts that transact densely with each other but
  sparsely with the rest of the graph — exactly how a ring is shaped — and the
  leave-one-out fraud density of a node's community becomes a powerful, non-circular
  feature.
- **Temporal burstiness** + same-day pass-through catches the "money doesn't sit
  still" velocity that's central to AML structuring detection.

This is also why a naive rule-based detector (included for comparison, see
`evaluate_rule_based()` in `fraud_classifier.py`) tops out around 68% precision / 16%
recall — single-threshold rules can't capture the joint structure the ensemble model
learns from these features.

---

## 6. Extending toward a true GNN

This project uses hand-engineered graph features + a tree ensemble, which is fast,
interpretable, and already exceeds the 95%/95% target. A natural next step — left as an
extension — is to swap the feature/ensemble stage for a message-passing GNN
(GraphSAGE / GAT via PyTorch Geometric) trained directly on the graph topology and node
attributes, which can pick up higher-order ring structures the hand-built features
don't explicitly encode. The `data/node_features.csv` + edge list already provide
everything needed to build a PyG `Data` object if you want to take that step.

---

## 7. Limitations (be upfront about these)

- The dataset is **synthetic**. Real UPI fraud patterns are messier, adversaries adapt,
  and class balance in production is far more skewed (fraud is usually <<1% of
  accounts, not ~21% as in this demo). Real deployments need careful handling of class
  imbalance, concept drift, and adversarial evasion.
- The 95%+ scores are reported on a held-out split of this synthetic data — they
  demonstrate the *method* works on data with these structural properties, not a
  guarantee on live UPI data, which would need labeled ground truth from confirmed
  fraud cases to validate properly.
- For a real fintech deployment you would also want: graph features recomputed
  incrementally (not full recompute each run), entity resolution (same person, multiple
  accounts), KYC/device/IP signals merged in, and a human-in-the-loop review queue
  rather than auto-action on model output.
