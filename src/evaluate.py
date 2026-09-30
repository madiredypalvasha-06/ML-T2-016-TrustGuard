"""Evaluation orchestration shared by the experiment scripts."""
import numpy as np
import torch


def _loader(ds, batch_size=256):
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


from src import config, metrics as mt, uncertainty as uc


def score_tensor(model, x, temperature=1.0, device="cpu", do_mc=False,
                 n_mc=config.MC_DROPOUT_PASSES, max_mc=None):
    """Run the model over a raw tensor `x` and produce probs + uncertainty scores.

    Returns:
      probs  : (N, C) float array (post temperature scaling, mean MC if do_mc)
      labels : (N,)  int array (all zeros if not provided properly; caller sets)
      scores : dict of scalar uncertainty scores (N,)
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    subset = None
    if do_mc and max_mc is not None and len(x) > max_mc:
        rng = np.random.RandomState(config.SEED)
        subset = np.sort(rng.choice(len(x), max_mc, replace=False))
        x_eff = x[subset]
    else:
        x_eff = x

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_eff, torch.zeros(len(x_eff), dtype=torch.long)),
        batch_size=256, shuffle=False,
    )

    if do_mc:
        logits_all, _ = uc.mc_dropout_logits(model, loader, device, passes=n_mc)
        mc_stats = uc.mc_dropout_stats(logits_all, temperature=temperature)
        probs = mc_stats["probs_mean"]
    else:
        logits = uc.predict_logits(model, loader, device)
        probs = uc.probs_from_logits(logits / temperature).numpy()
        mc_stats = None

    scores = uc.make_uncertainty_scores(
        probs, mc_stats=mc_stats,
    )
    return probs, scores, subset


def score_dataset(model, loader, temperature=1.0, device="cpu", do_mc=False,
                  n_mc=config.MC_DROPOUT_PASSES, max_mc=None):
    """Score any (x, y) loader; returns probs, labels, scores aligned on the
    same instances."""
    x_all, y_all = [], []
    for xb, yb in loader:
        x_all.append(xb.cpu())
        y_all.append(yb.cpu())
    x = torch.cat(x_all)
    y = torch.cat(y_all).numpy()
    probs, scores, subset = score_tensor(model, x, temperature, device, do_mc, n_mc, max_mc)
    return probs, y, scores, subset


def summarize(probs, labels, scores):
    """Compute all headline metrics for a scored set."""
    n_classes = probs.shape[1]
    correct = mt.correct_mask(probs, labels)
    out = {
        "accuracy": mt.accuracy(probs, labels),
        "brier": mt.brier_score(probs, labels, n_classes),
        "n_examples": int(len(labels)),
    }
    out["ece_softmax"] = mt.expected_calibration_error(probs, labels)["ece"]
    for name, unc in scores.items():
        if name.startswith("msp_"):
            continue
        out[f"error_auroc_{name}"] = mt.error_detection_auroc(unc, correct)
    return out


def score_and_fit_temperature(model, cal_loader, device="cpu"):
    """Fit temperature scaling on the calibration loader."""
    logits_cal = uc.predict_logits(model, cal_loader, device)
    labels_cal = []
    for _, yb in cal_loader:
        labels_cal.append(yb.cpu())
    labels_cal = torch.cat(labels_cal)
    ts = uc.TemperatureScaling()
    temp = ts.fit(logits_cal, labels_cal)
    return ts, float(temp)


def evaluate_clean(data, model, device):
    """Full clean-set run: fit temperature, score test set with all methods.

    `data` = (train_ds, calibration_ds, test_ds). Returns results dict.
    """
    _, cal_ds, test_ds = data
    cal_loader = _loader(cal_ds)
    test_loader = _loader(test_ds)

    ts, temp = score_and_fit_temperature(model, cal_loader, device)
    probs, labels, scores, _ = score_dataset(model, test_loader, temp, device)
    probs_raw, _, _, _ = score_dataset(model, test_loader, 1.0, device)
    mc_probs, mc_labels, mc_scores, _ = score_dataset(
        model, test_loader, temp, device, do_mc=True
    )
    assert (mc_labels == labels).all()
    scores.update({k: v for k, v in mc_scores.items()
                   if k.startswith("mc_")})

    summary = summarize(probs, labels, scores)
    summary["temperature"] = temp
    coverage = mt.risk_coverage_summary(probs, labels, scores)
    return {
        "probs": probs,
        "probs_raw": probs_raw,
        "labels": labels,
        "scores": scores,
        "summary": summary,
        "coverage": coverage,
        "temperature": temp,
        "ts": ts,
    }