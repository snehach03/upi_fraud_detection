"""
One-shot pipeline runner: generate data -> extract graph features ->
train classifier -> export dashboard JSON.

Usage: python run_pipeline.py
Then open app/index.html in a browser, or serve it: python -m http.server --directory app 8000
"""
import subprocess, sys, os

STEPS = [
    ("Generating synthetic UPI transaction network", "src/data_generator.py"),
    ("Extracting graph features (NetworkX + Louvain)", "src/graph_features.py"),
    ("Training fraud ring classifier", "src/fraud_classifier.py"),
    ("Exporting dashboard data", "src/export_for_dashboard.py"),
]

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    for label, script in STEPS:
        print(f"\n{'='*70}\n>>> {label}\n{'='*70}")
        result = subprocess.run([sys.executable, os.path.join(root, script)], cwd=os.path.join(root, "src"))
        if result.returncode != 0:
            print(f"Step failed: {script}")
            sys.exit(1)
    print("\nPipeline complete. Open app/index.html in a browser, or run:")
    print("  python -m http.server --directory app 8000")
    print("  then visit http://localhost:8000")
