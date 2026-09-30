"""02 — Baseline evaluation on clean CIFAR-10 test set.

Fits the temperature on the calibration split, then evaluates all uncertainty
methods (softmax, temperature-scaled, MC Dropout, entropy) on the held-out test
set: accuracy, ECE, Brier, error-detection AUROC, reliability diagrams and
risk-coverage curves.

Usage: python scripts/02_evaluate_clean.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np

from src import config, metrics as mt 
from src.data import get_clean_datasets
from src.evaluate import evaluate_clean  
from src.models import load_model

METHOD_LABELS = {
    "softmax": "A. Softmax (MSP)",
    "entropy": "D. Entropy",
    "mc_entropy": "B. MC Dropout (entropy)",
    "mc_variance": "B. MC Dropout (variance)",
    "mc_mutual_info": "B. MC Dropout (mutual info)",
}


def main():
    device = config.get_device()
    print(f"[clean-eval] device: {device}")
    data = get_clean_datasets()
    model = load_model(config.MODEL_CHECKPOINT, device)
    r = evaluate_clean(data, model, device)

    # save temperature params
    ts = r["ts"]
    ts.save(ts.temperature, config.TEMP_PARAMS)
    print(f"[clean-eval] temperature = {r['temperature']:.4f}")

    # figures
    probs, labels, scores = r["probs"], r["labels"], r["scores"]

    # 1) reliability diagram for raw vs temperature-scaled probs
    mt.reliability_diagram(
        probs, labels, os.path.join(config.FIGURE_DIR, "reliability_temp_scaled.png"),
        title="Temperature-scaled softmax (test set)",
    )
    mt.reliability_diagram(
        r["probs_raw"], labels,
        os.path.join(config.FIGURE_DIR, "reliability_raw_softmax.png"),
        title="Raw softmax before temperature scaling (test set)",
    )

    # 2) risk-coverage across methods
    mc_scores = {k: v for k, v in scores.items() if k.startswith("mc_")}
    plot_scores = {"softmax": scores["softmax"], "entropy": scores["entropy"],
                   **mc_scores}
    mt.plot_risk_coverage(
        probs, labels, plot_scores, METHOD_LABELS,
        os.path.join(config.FIGURE_DIR, "risk_coverage_clean.png"),
        title="Risk-coverage on clean test set (lower is better)",
    )

    # 3) summary + coverage table
    summary = r["summary"]
    labels_for_table = {k: METHOD_LABELS.get(k, k) for k in plot_scores}
    cov_table = {}
    for name, row in r["coverage"].items():
        if name in plot_scores:
            cov_table[labels_for_table[name]] = {
                f"cov_{int(c*100)}": round(acc, 4) for c, acc in row.items()
            }

    print("\n=== Clean test summary ===")
    print(f"accuracy:            {summary['accuracy']:.4f}")
    print(f"Brier (temp-scaled): {summary['brier']:.4f}")
    print(f"ECE (temp-scaled):   {summary['ece_softmax']:.4f}")
    print("\nError-detection AUROC (higher = uncertainty tracks mistakes better):")
    for k, v in summary.items():
        if k.startswith("error_auroc"):
            short = k.replace("error_auroc_", "")
            print(f"  {METHOD_LABELS.get(short, short):30s}: {v:.4f}")

    print("\nAccuracy at coverage (clean test):")
    header = "method".ljust(30) + "  ".join(f"{c*100:.0f}%" for c in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5))
    print("  " + header)
    for name, vals in cov_table.items():
        cells = "  ".join(f"{vals.get(f'cov_{c*100:.0f}', 0):.4f}" for c in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5))
        print(f"  {name.ljust(30)}{cells}")

    out = {
        "summary": summary,
        "coverage_table": cov_table,
        "temperature": r["temperature"],
    }
    with open(os.path.join(config.RESULT_DIR, "clean_eval.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\n[clean-eval] saved -> results/clean_eval.json")


if __name__ == "__main__":
    main()