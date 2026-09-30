"""03 — Distribution-shift experiments with CIFAR-10-C.

1. Accuracy drop + confidence/ECE behaviour across corruption severities.
2. IN (clean) vs OUT (corrupted) separation AUROC for each uncertainty method.
3. Risk-coverage analysis under strong shift + abstention effect.

Usage: python scripts/03_shift_experiments.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import torch

from src import config, metrics as mt
from src.data import get_clean_datasets, get_shift_data
from src.evaluate import score_and_fit_temperature, score_tensor
from src.models import load_model

SEVS = [1, 3, 5]
METHOD_LABELS = {
    "softmax": "A. Softmax (MSP)",
    "entropy": "D. Entropy",
    "mc_entropy": "B. MC Dropout (entropy)",
    "mc_variance": "B. MC Dropout (variance)",
}


def run():
    device = config.get_device()
    print(f"[shift] device: {device}")
    _, cal_ds, test_ds = get_clean_datasets()
    model = load_model(config.MODEL_CHECKPOINT, device)
    ts, temp = score_and_fit_temperature(
        model, torch.utils.data.DataLoader(cal_ds, batch_size=512, shuffle=False), device
    )
    print(f"[shift] fitted temperature = {temp:.4f}")

    # ---- clean reference scores (full test, 10k) ----
    x_clean = torch.stack([t for t, _ in test_ds])
    y_clean = torch.tensor([lbl for _, lbl in test_ds]).numpy()
    det_clean_scores = score_tensor(model, x_clean, temp, device, do_mc=False)[1]
    clean_probs_mc, clean_mc_scores, clean_mc_idx = score_tensor(
        model, x_clean, temp, device, do_mc=True, max_mc=3000
    )

    rows = []
    for corr in config.CORRUPTIONS:
        for sev in SEVS:
            xc, yc = get_shift_data(corr, sev, device=device)

            det_probs, det_scores, _ = score_tensor(model, xc, temp, device)
            mc_probs, mc_scores, mc_idx = score_tensor(
                model, xc, temp, device, do_mc=True, max_mc=2500
            )

            acc = mt.accuracy(det_probs, yc)
            ece = mt.expected_calibration_error(det_probs, yc)["ece"]
            conf = float(det_probs.max(axis=-1).mean())
            correct_det = mt.correct_mask(det_probs, yc)

            err_auroc = {}
            for name, unc in det_scores.items():
                err_auroc[name] = mt.error_detection_auroc(unc, correct_det)
            # MC ran on the subset `mc_idx` -> align correctness to that subset
            correct_mc = mt.correct_mask(mc_probs, yc[mc_idx]) if mc_idx is not None \
                else correct_det
            for name in ("mc_entropy", "mc_variance"):
                err_auroc[name] = mt.error_detection_auroc(
                    mc_scores[name], correct_mc
                )

            rows.append({
                "corruption": corr, "severity": sev,
                "accuracy": acc, "ece": ece, "mean_conf": conf,
                "err_auroc": err_auroc,
            })
            print(f"[shift] {corr:16s} sev{sev}  acc={acc:.4f} "
                  f"ece={ece:.4f} conf={conf:.4f}")

    # ---- AUROC: clean (IN) vs corrupted (OUT) separation ----
    n = 3000   
    rng = np.random.RandomState(config.SEED)
    in_idx = rng.choice(len(x_clean), n, replace=False)
    out_blocks = {
        "gaussian_noise_sev5": get_shift_data("gaussian_noise", 5, device=device)[0][:n],
        "contrast_sev5": get_shift_data("contrast", 5, device=device)[0][:n],
    }

    # deterministic methods: score IN with same sub-sampled indices
    auroc_tbl = {}
    for out_name, xout in out_blocks.items():
        out_det_scores = score_tensor(model, xout, temp, device)[1]
        out_mc_scores = score_tensor(model, xout, temp, device,
                                     do_mc=True, max_mc=n)[1]
        for name in ("softmax", "entropy"):
            auroc = mt.auroc_score(-det_clean_scores[name][in_idx],
                                   -out_det_scores[name])
            auroc_tbl[f"{name}@{out_name}"] = auroc
        for name in ("mc_entropy", "mc_variance"):
            auroc = mt.auroc_score(-clean_mc_scores[name],
                                   -out_mc_scores[name])
            auroc_tbl[f"{name}@{out_name}"] = auroc
        print(f"[shift] AUROC ({out_name}): "
              + ", ".join(f"{k}={v:.3f}" for k, v in auroc_tbl.items()
                          if k.endswith(f"@{out_name}")))

    # ---- risk-coverage under strong shift + abstention effect ----
    abs_effect = {}
    for label, (corr, sev) in {"gaussian_noise sev5": ("gaussian_noise", 5),
                               "fog sev5": ("fog", 5)}.items():
        xo, yo = get_shift_data(corr, sev, device=device)
        det_probs_o, det_scores_o, _ = score_tensor(model, xo, temp, device)
        mc_probs_o, mc_scores_o, _ = score_tensor(
            model, xo, temp, device, do_mc=True, max_mc=2500
        )
        plot_scores = {
            "softmax": det_scores_o["softmax"],
            "entropy": det_scores_o["entropy"],
            "mc_entropy": mc_scores_o["mc_entropy"],
            "mc_variance": mc_scores_o["mc_variance"],
        }
        mt.plot_risk_coverage(
            det_probs_o, yo, plot_scores, METHOD_LABELS,
            os.path.join(config.FIGURE_DIR,
                         f"risk_coverage_shift_{corr}_sev{sev}.png"),
            title=f"Risk-coverage under {label} (lower is better)",
        )
        abs_effect[label] = {}
        for name, unc in plot_scores.items():
            acc_full = mt.accuracy(det_probs_o, yo)
            acc_cov, _ = mt.accuracy_at_coverage(det_probs_o, yo, unc, 0.9)
            abs_effect[label][name] = {
                "full_acc": acc_full, "acc_at_90": acc_cov,
                "risk_reduction": acc_full - acc_cov,
            }
            print(f"[shift] {label:22s} "
                  f"{METHOD_LABELS[name]:30s} full_acc={acc_full:.4f}  "
                  f"acc@90%cov={acc_cov:.4f}")

    tab = {}
    for r in rows:
        key = f"{r['corruption']}_sev{r['severity']}"
        tab[key] = {
            "accuracy": round(r["accuracy"], 4),
            "ece": round(r["ece"], 4),
            "mean_conf": round(r["mean_conf"], 4),
            "err_auroc_softmax": round(r["err_auroc"]["softmax"], 4),
            "err_auroc_entropy": round(r["err_auroc"]["entropy"], 4),
            "err_auroc_mc_entropy": round(r["err_auroc"]["mc_entropy"], 4),
        }
    out = {"rows": rows, "auroc_table": auroc_tbl, "summary_table": tab,
           "abs_effect": abs_effect, "temperature": temp}
    with open(os.path.join(config.RESULT_DIR, "shift_eval.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("[shift] saved -> results/shift_eval.json")


if __name__ == "__main__":
    run()