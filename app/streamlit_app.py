"""ML-T2-016 interactive demo — "When NOT to trust the prediction".

Run:  streamlit run app/streamlit_app.py

Lets you feed an image (clean CIFAR-10 sample, corrupted/shifted sample, or
your own upload) and see:
  * the predicted class
  * temperature-scaled confidence
  * MC-Dropout predictive uncertainty
  * an evidence-based TRUST / REVIEW / ABSTAIN decision
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src import config, corruptions
from src.data import get_clean_datasets
from src.models import load_model
from src import uncertainty as uc

st.set_page_config(page_title="ML-T2-016 · Trust-aware classifier", layout="wide")

CLASSES = config.CLASS_NAMES


@st.cache_resource
def load_resources():
    device = config.get_device()
    model = load_model(config.MODEL_CHECKPOINT, device)
    model.eval()
    temperature = 1.0
    if os.path.exists(config.TEMP_PARAMS):
        ts = uc.TemperatureScaling.load(config.TEMP_PARAMS)
        temperature = float(ts.temperature)
    elif os.path.exists(os.path.join(config.RESULT_DIR, "clean_eval.json")):
        with open(os.path.join(config.RESULT_DIR, "clean_eval.json")) as f:
            temperature = json.load(f)["temperature"]
    _, cal_ds, _ = get_clean_datasets()
    return model, device, temperature, cal_ds


@st.cache_data
def fit_trust_threshold(device_name, coverage_target, n_cal=2000):
    """Predictive-entropy value at `coverage_target` (fraction kept) computed on
    the calibration split. Evidence-based abstention rule."""
    model, device, temperature, cal_ds = load_resources()
    rng = np.random.RandomState(config.SEED)
    idx = rng.choice(len(cal_ds), n_cal, replace=False)
    xs = torch.stack([cal_ds[i][0] for i in idx]).to(device)
    logits_all, _ = uc.mc_dropout_logits(
        model, torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xs, torch.zeros(len(xs), dtype=torch.long)),
            batch_size=256, shuffle=False),
        device,
    )
    stats = uc.mc_dropout_stats(logits_all, temperature=temperature)
    entropy = stats["entropy"]
    k = max(1, int(coverage_target * n_cal))
    thr = np.sort(entropy)[k - 1]
    return float(thr)


def predict_single(model, device, temperature, img):
    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
    probs = torch.softmax(logits / temperature, dim=-1)[0].cpu().numpy()
    conf = float(probs.max())
    pred = int(probs.argmax())
    det_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    # MC dropout: stochastic passes on the (possibly padded) input
    logits_all = crop_mc_logits(model, device, img.unsqueeze(0), temperature)
    stats = uc.mc_dropout_stats(logits_all, temperature=temperature)
    mc_probs = np.asarray(stats["probs_mean"]).squeeze()
    mc_entropy = float(np.asarray(stats["entropy"]).squeeze())
    mc_variance = float(np.asarray(stats["variance"]).squeeze())
    mc_conf = float(mc_probs.max())
    mc_pred = int(mc_probs.argmax())
    return {
        "pred": pred, "conf": conf, "entropy": det_entropy,
        "mc_pred": mc_pred, "mc_conf": mc_conf,
        "mc_entropy": mc_entropy, "mc_variance": mc_variance,
        "probs": probs,
    }


@torch.no_grad()
def crop_mc_logits(model, device, x_, temperature):
    model.train()
    model.apply(lambda m: isinstance(m, torch.nn.Dropout) and m.train())
    outs = []
    xb = x_.to(device)
    for _ in range(config.MC_DROPOUT_PASSES):
        outs.append(model(xb).detach().cpu())
    return torch.stack(outs)


def color_emoji(decision):
    return {"TRUST": " TRUST", "REVIEW": " REVIEW",
            "ABSTAIN": " ABSTAIN"}[decision]


def main():
    st.title("Teaching an ML system when **not** to trust its prediction")
    st.caption("Problem ID ML-T2-016 · Track 2 Advanced ML Internship · Stage 2 (Development)")

    model, device, temperature, cal_ds = load_resources()
    _, _, test_ds = get_clean_datasets()

    st.sidebar.header("Input")
    source = st.sidebar.radio("Image source", [
        "Clean CIFAR-10 sample", "Shifted (corrupted) sample", "Upload image",
    ])
    if source == "Clean CIFAR-10 sample":
        idx = st.sidebar.slider("Test image index", 0, len(test_ds) - 1, 100)
        img = test_ds[idx][0]
        actual = int(test_ds[idx][1])
        caption = f"Clean test image #{idx} · true class: **{CLASSES[actual]}**"
    elif source == "Shifted (corrupted) sample":
        corr = st.sidebar.selectbox("Corruption", corruptions.available_corruptions())
        sev = st.sidebar.slider("Severity", 1, 5, 3)
        idx = st.sidebar.slider("Test image index", 0, len(test_ds) - 1, 100)
        img = corruptions.corrupt(test_ds[idx][0].unsqueeze(0), corr, sev)[0].cpu()
        actual = int(test_ds[idx][1])
        caption = f"{corr} (severity {sev}) applied to test image #{idx} · true class: **{CLASSES[actual]}**"
    else:
        up = st.sidebar.file_uploader("Upload an image (will be resized to 32×32)",
                                      type=["png", "jpg", "jpeg", "bmp"])
        if up is None:
            st.info("Upload an image to get started.")
            st.stop()
        pil = Image.open(up).convert("RGB").resize((32, 32))
        img = torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 255.0
        actual = -1
        caption = "Uploaded image (resized to 32×32)"

    model.eval()
    result = predict_single(model, device, temperature, img)

    coverage_target = st.sidebar.slider(
        "Abstention threshold: keep the most-confident % of predictions",
        min_value=50, max_value=100, value=90, step=5, key=None,
    )
    thr = fit_trust_threshold(device, coverage_target / 100.0)

    pred = result["mc_pred"]
    untrusted = result["mc_entropy"] > thr

    st.sidebar.header("Abstention rule")
    st.sidebar.markdown(
        f"MC predictive entropy threshold at {coverage_target}% coverage: "
        f"**{thr:.4f}**"
    )
    st.sidebar.markdown(
        f"> Entropy of this image: **{result['mc_entropy']:.4f}** → "
        f"{'ABOVE (review)' if untrusted else 'below (trust)'} threshold."
    )

    c1, c2 = st.columns([1.2, 2])
    with c1:
        st.image(img.permute(1, 2, 0).numpy(), caption=caption, width="stretch")

    with c2:
        decision = "TRUST" if result["mc_entropy"] <= thr else "REVIEW"
        st.subheader(color_emoji(decision) + " prediction")
        st.markdown(f"**Predicted class:** {CLASSES[pred]}")
        st.markdown(f"**Temperature-scaled confidence:** {result['conf']:.1%}")
        st.markdown(f"**MC-Dropout confidence:** {result['mc_conf']:.1%}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Deterministic entropy", f"{result['entropy']:.4f}")
        m2.metric("MC predictive entropy", f"{result['mc_entropy']:.4f}")
        m3.metric("MC probability variance", f"{result['mc_variance']:.6f}")

        if actual >= 0:
            is_correct = pred == actual
            st.markdown(
                f"True class **{CLASSES[actual]}** — prediction is "
                f"{' correct' if is_correct else ' wrong'}"
            )

        st.markdown("**Per-class probabilities (temp-scaled):**")
        import pandas as pd
        chart = pd.DataFrame(
            {"probability": result["probs"]}, index=CLASSES
        )
        st.bar_chart(chart)

        st.caption(
            "Trust rule: keep a prediction when its MC predictive entropy is "
            f"below the {coverage_target}% coverage threshold ({thr:.4f}) "
            "fitted on the calibration split. Above it, flag for review."
        )


if __name__ == "__main__":
    main()