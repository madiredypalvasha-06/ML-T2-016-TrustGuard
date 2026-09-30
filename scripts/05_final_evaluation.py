"""05 — Final unified evaluation across every reliability method.

Assembles all candidate approaches from the research brief:
  A  softmax confidence + temperature scaling
  B  Monte-Carlo Dropout
  C  deep ensemble (disagreement / entropy)
  D  feature-space Mahalanobis distance

Reports on the held-out clean test and controlled corruptions, then evaluates
the deployable `TrustAwareClassifier` (fused verdict) end-to-end.

Usage: python scripts/05_final_evaluation.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import torch

from src import config, ensembles as ens, mahalanobis as mh
from src import metrics as mt
from src.data import get_clean_datasets, get_shift_data
from src.evaluate import score_and_fit_temperature, score_tensor
from src.models import load_model
from src import uncertainty as uc

SEVS = [1, 3, 5]
METHODS = [
    "softmax", "entropy", "mc_entropy", "ens_entropy",
    "ens_fraction_disagree", "ens_variance", "mahalanobis",
]
METHOD_LABELS = {
    "softmax": "A. Softmax (MSP)",
    "entropy": "D. entropy",
    "mc_entropy": "B. MC Dropout",
    "ens_entropy": "C. Ensemble",
    "ens_variance": "C. Ensemble (variance)",
    "ens_fraction_disagree": "C. Ensemble (disagreement)",
    "mahalanobis": "D. Mahalanobis distance",
}


def features_for(model, x, device, batch_size=256):
    """Penultimate features (N, 256)."""
    outs = []
    for i in range(0, len(x), batch_size):
        f, _ = model.forward_features(x[i:i + batch_size].to(device))
        outs.append(f.cpu().numpy())
    return np.concatenate(outs)


def main(members=config.ENSEMBLE_MEMBERS, out_path=None):
    device = config.get_device()
    print(f"[final-eval] device: {device}", flush=True)
    _, cal_ds, test_ds = get_clean_datasets()

    base = load_model(config.MODEL_CHECKPOINT, device)
    tags = [f"member_{i}" for i in range(members)]
    members_l = ens.load_members(tags=tags, device=device)
    print(f"[final-eval] loaded baseline + {len(members_l)} ensemble members",
          flush=True)

    cal_x = torch.stack([t for t, _ in cal_ds])
    cal_y = torch.tensor([lbl for _, lbl in cal_ds]).numpy()
    test_x = torch.stack([t for t, _ in test_ds]).cpu()
    test_y = torch.tensor([lbl for _, lbl in test_ds]).numpy()

    # ---- temperature & mahalanobis fitting (calibration split only) ----
    _, temp = score_and_fit_temperature(
        base, torch.utils.data.DataLoader(cal_ds, batch_size=512, shuffle=False),
        device)
    print(f"[final-eval] temperature = {temp:.4f}", flush=True)
    cal_feats = features_for(base, cal_x.to(device), device)
    maha = mh.MahalanobisOOD().fit(cal_feats, cal_y)
    maha.save(config.MAHALANOBIS_PATH)

    # ---- clean test scoring: all methods ----
    probs_base, scores_base, _ = score_tensor(base, test_x.to(device), temp, device)
    _, mc_scores, _ = score_tensor(
        base, test_x.to(device), temp, device, do_mc=True, max_mc=10000)
    ens_res = ens.ensemble_predict_loader(
        members_l, torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_x, torch.from_numpy(test_y)),
            batch_size=256, shuffle=False), temp, device)
    ens_probs, ens_scores = ens_res["probs"], ens_res["scores"]
    maha_dist = maha.distance(features_for(base, test_x.to(device), device))

    scores = {
        "softmax": scores_base["softmax"],
        "entropy": scores_base["entropy"],
        "mc_entropy": mc_scores["mc_entropy"],
        "ens_entropy": ens_scores["ens_entropy"],
        "ens_variance": ens_scores["ens_variance"],
        "ens_fraction_disagree": ens_scores["ens_fraction_disagree"],
        "mahalanobis": maha_dist,
    }

    clean = {}
    clean["accuracy_base"] = mt.accuracy(probs_base, test_y)
    clean["accuracy_ensemble"] = mt.accuracy(ens_probs, test_y)
    clean["ece_base"] = mt.expected_calibration_error(probs_base, test_y)["ece"]
    clean["ece_ensemble"] = mt.expected_calibration_error(ens_probs, test_y)["ece"]
    clean["brier_base"] = mt.brier_score(probs_base, test_y)
    clean["brier_ensemble"] = mt.brier_score(ens_probs, test_y)
    clean["error_auroc"] = {}
    correct_ens = mt.correct_mask(ens_probs, test_y)
    correct_base = mt.correct_mask(probs_base, test_y)
    for name in METHODS:
        if name in ("softmax", "entropy"):
            clean["error_auroc"][name] = mt.error_detection_auroc(
                scores[name], correct_base)
        else:
            clean["error_auroc"][name] = mt.error_detection_auroc(
                scores[name], correct_ens)
    print("[final-eval] clean: base_acc={:.4f} ens_acc={:.4f} "
          "ece_base={:.4f} ece_ens={:.4f}".format(
              clean["accuracy_base"], clean["accuracy_ensemble"],
              clean["ece_base"], clean["ece_ensemble"]), flush=True)

    # coverage table using ensemble predictions for ens/maha, base for others
    cov_probs = {**{k: probs_base for k in ("softmax", "entropy")},
                 **{k: ens_probs for k in ("mc_entropy", "ens_entropy",
                                           "ens_variance",
                                           "ens_fraction_disagree",
                                           "mahalanobis")}}
    clean["coverage"] = {}
    for name in METHODS:
        row = {}
        for cov in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5):
            acc, _ = mt.accuracy_at_coverage(cov_probs[name], test_y,
                                             scores[name], cov)
            row[f"cov_{int(cov*100):d}"] = acc
        clean["coverage"][name] = row

    # ---- shift (accuracy/ECE/conf + error-AUROC) ----
    shift_rows = []
    for corr in config.CORRUPTIONS:
        for sev in SEVS:
            xs, ys = get_shift_data(corr, sev, device=device)
            p_base, _, _ = score_tensor(base, xs, temp, device)
            e_res = ens.ensemble_predict_loader(
                members_l, torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(xs.cpu(), torch.from_numpy(ys)),
                    batch_size=256, shuffle=False), temp, device)
            ep, es = e_res["probs"], e_res["scores"]
            maha_s = maha.distance(features_for(base, xs, device))
            correct_b = mt.correct_mask(p_base, ys)
            correct_e = mt.correct_mask(ep, ys)
            row = {
                "corruption": corr, "severity": sev,
                "acc_base": float(mt.accuracy(p_base, ys)),
                "acc_ensemble": float(mt.accuracy(ep, ys)),
                "ece_base": float(mt.expected_calibration_error(p_base, ys)["ece"]),
                "ece_ensemble": float(mt.expected_calibration_error(ep, ys)["ece"]),
                "mean_conf_base": float(p_base.max(axis=-1).mean()),
            }
            # error-detection AUROCs on this shift for each score
            r = row["err_auroc"] = {}
            r["softmax"] = mt.error_detection_auroc(1 - p_base.max(-1), correct_b)
            r["entropy"] = mt.error_detection_auroc(scores_base["entropy"], correct_b)
            r["ens_entropy"] = mt.error_detection_auroc(es["ens_entropy"], correct_e)
            r["ens_disagree"] = mt.error_detection_auroc(es["ens_fraction_disagree"], correct_e)
            r["mahalanobis"] = mt.error_detection_auroc(maha_s, correct_e)
            shift_rows.append(row)
            print("[final-eval] {} sev{}  acc_b={:.4f} acc_e={:.4f} "
                  "conf={:.3f}".format(corr, sev, row["acc_base"],
                                       row["acc_ensemble"], row["mean_conf_base"]),
                  flush=True)

    # ---- IN-vs-OUT AUROC across all methods ----
    n = 3000
    rng = np.random.RandomState(config.SEED)
    in_idx = rng.choice(len(test_x), n, replace=False)
    out_blocks = {
        "gaussian_noise_sev5": get_shift_data("gaussian_noise", 5, device)[0][:n],
        "contrast_sev5": get_shift_data("contrast", 5, device)[0][:n],
    }
    in_scores = {k: v[in_idx] for k, v in scores.items()}
    auroc_tbl = {}
    for oname, xout in out_blocks.items():
        o_s, o_b = {}, {}
        p_o, s_o, _ = score_tensor(base, xout, temp, device)
        e_o = ens.ensemble_predict_loader(
            members_l, torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(xout.cpu(),
                                               torch.zeros(len(xout), dtype=torch.long)),
                batch_size=256, shuffle=False), temp, device)
        m_o = maha.distance(features_for(base, xout, device))
        o_b = {"softmax": s_o["softmax"], "entropy": s_o["entropy"]}
        o_e = {k: e_o["scores"][k] for k in
               ("ens_entropy", "ens_variance", "ens_fraction_disagree")}
        for name in ("softmax", "entropy"):
            auroc_tbl[f"{name}@{oname}"] = mt.auroc_score(
                -in_scores[name], -o_b[name])
        # recompute MC dropout on the OUT block
        o_mc = score_tensor(base, xout, temp, device, do_mc=True, max_mc=n)[1]
        auroc_tbl[f"mc_entropy@{oname}"] = mt.auroc_score(
            -in_scores["mc_entropy"], -o_mc["mc_entropy"])
        for name in ("ens_entropy", "ens_variance", "ens_fraction_disagree"):
            auroc_tbl[f"{name}@{oname}"] = mt.auroc_score(
                -in_scores[name], -o_e[name])
        auroc_tbl[f"mahalanobis@{oname}"] = mt.auroc_score(
            -in_scores["mahalanobis"], -m_o)
    print("[final-eval] IN-vs-OUT AUROC:", flush=True)
    for k in sorted(auroc_tbl):
        print(f"    {k:42s} {auroc_tbl[k]:.4f}", flush=True)

    # ---- risk-coverage figures + abstention effect ----
    rc_scores = {k: scores[k] for k in
                 ("softmax", "entropy", "mc_entropy", "ens_entropy",
                  "ens_fraction_disagree", "mahalanobis")}
    cov_probs_rc = {k: ens_probs if k in ("mc_entropy", "ens_entropy",
                                          "ens_fraction_disagree", "mahalanobis")
                    else probs_base for k in rc_scores}
    mt.plot_risk_coverage(
        probs_base, test_y, {k: v for k, v in rc_scores.items()},
        METHOD_LABELS,
        os.path.join(config.FIGURE_DIR, "final_risk_coverage_clean.png"),
        title="Final: risk-coverage on clean test (lower is better)")

    abstain = {}
    for label, corr in {"gaussian_noise sev5": "gaussian_noise",
                        "fog sev5": "fog"}.items():
        xs, ys = get_shift_data(corr, 5, device=device)
        p_s, s_s, _ = score_tensor(base, xs, temp, device)
        e_s = ens.ensemble_predict_loader(
            members_l, torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(xs.cpu(),
                                               torch.from_numpy(ys)),
                batch_size=256, shuffle=False), temp, device)
        m_s = maha.distance(features_for(base, xs, device))
        rc = {"softmax": s_s["softmax"],
              "ens_entropy": e_s["scores"]["ens_entropy"],
              "ens_disagree": e_s["scores"]["ens_fraction_disagree"],
              "mahalanobis": m_s}
        rc_probs = {"ens_entropy": e_s["probs"], "ens_disagree": e_s["probs"],
                    "mahalanobis": e_s["probs"], "softmax": p_s}
        mt.plot_risk_coverage(
            p_s, ys, rc, METHOD_LABELS,
            os.path.join(config.FIGURE_DIR, f"final_risk_coverage_{corr}_sev5.png"),
            title=f"Final: risk-coverage under {label} (lower is better)")
        abstain[label] = {}
        for name, unc in rc.items():
            acc_full = mt.accuracy(rc_probs[name], ys)
            acc_cov, _ = mt.accuracy_at_coverage(rc_probs[name], ys, unc, 0.9)
            abstain[label][name] = {
                "full_acc": acc_full, "acc_at_90": acc_cov,
                "improvement": float(acc_cov - acc_full),
            }
        print(f"[final-eval] abstain {label}: " + ", ".join(
            f"{k}={v['acc_at_90']:.4f} (full {v['full_acc']:.4f})"
            for k, v in abstain[label].items()), flush=True)

    # ---- deployable verdict: fit thresholds on calibration, test on clean+shift ----
    tc = ens.TrustAwareClassifier(members_l, temperature=temp, coverage=0.9,
                                  device=device, mahalanobis=maha)
    tc.fit_thresholds(cal_x, cal_y)
    verdicts = {"clean": _evaluate_verdicts(tc, test_x, test_y, device),
                "gaussian_noise_sev5": _evaluate_verdicts(
                    tc, get_shift_data("gaussian_noise", 5, device)[0].cpu(),
                    test_y, device),
                "fog_sev5": _evaluate_verdicts(
                    tc, get_shift_data("fog", 5, device)[0].cpu(), test_y, device)}
    for k, v in verdicts.items():
        print("[final-eval] verdict {}: {} (acc on trusted: {:.4f})".format(
            k, v["fraction_trusted"], v.get("acc_on_trusted", float("nan"))),
            flush=True)

    out = {
        "clean": clean,
        "shift_rows": shift_rows,
        "auroc_table": auroc_tbl,
        "abstain": abstain,
        "temperature": temp,
        "verdicts": verdicts,
        "method_labels": METHOD_LABELS,
    }
    if out_path is None:
        out_path = os.path.join(config.RESULT_DIR, "final_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[final-eval] saved -> {out_path}", flush=True)


def _evaluate_verdicts(tc, x, y, device):
    preds, verdicts, dists = [], [], []
    for i in range(0, len(x), 512):
        xb = x[i:i + 512]
        r = tc.predict_batch(xb.cpu() if xb.device.type == "cpu" else xb)
        preds.append(r["preds"])
        verdicts.append(r["verdicts"])
        dists.append(np.asarray(r["mahalanobis_dist"]))
    preds = np.concatenate(preds)
    verdicts = np.concatenate(verdicts)
    dists = np.concatenate(dists)
    trust_mask = verdicts == "TRUST"
    correct = preds == y
    return {
        "fraction_trusted": float(trust_mask.mean()),
        "coverage_reached": float(trust_mask.mean()),
        "acc_all": float(correct.mean()),
        "acc_on_trusted": float(correct[trust_mask].mean())
        if trust_mask.any() else float("nan"),
        "flagged_fraction": float((~trust_mask).mean()),
        "mean_mahalanobis_on_clean": float(dists.mean()),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--members", type=int, default=config.ENSEMBLE_MEMBERS,
                   help="how many ensemble members to load (for dry runs)")
    p.add_argument("--out", type=str, default=None,
                   help="output json path (default results/final_eval.json)")
    a = p.parse_args()
    main(members=a.members, out_path=a.out)