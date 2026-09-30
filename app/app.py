"""TrustGuard — a mission-control dashboard for ML-T2-016.

Four instrument panels:
  1. Mission Control .. probe a single input and read its trust verdict.
  2. Drift Monitor ..... push a corrupted feed through the pipeline and see
                         how much the system refuses.
  3. Calibration Lab ... inspect calibrated vs raw confidence behaviour.
  4. Method Showdown ... every reliability approach, side by side.

Run:  streamlit run app/app.py
"""
import io
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
import theme as th  # noqa: E402
from src import config  # noqa: E402

PAGES = ["Mission Control", "Drift Monitor", "Calibration Lab", "Method Showdown"]

st.set_page_config(page_title="TrustGuard · ML-T2-016",
                   page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown(th.CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# shared plot builders
# ---------------------------------------------------------------------------
def reliability_plot(probs, y, title, bins=15):
    preds = probs.argmax(-1)
    conf = probs.max(-1)
    correct = (preds == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers, acc, dens, ece = [], [], [], 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if i == bins - 1:
            m = (conf >= edges[i]) & (conf <= edges[i + 1])
        if not m.any():
            continue
        centers.append((edges[i] + edges[i + 1]) / 2)
        acc.append(correct[m].mean())
        dens.append(m.sum() / len(conf))
        ece += float(dens[-1] * abs(acc[-1] - centers[-1]))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=centers, y=acc, name="observed accuracy",
                         marker_color=th.SKY,
                         marker_line=dict(color=th.PANEL2, width=1),
                         hovertemplate="bin %.2f<br>acc %.3f<br>density %.3f"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color=th.MUTED, dash="dot", width=1.5),
                             name="perfect"))
    fig.add_annotation(text=f"ECE = {ece:.4f}", xref="paper", yref="paper",
                       x=0.98, y=0.06, xanchor="right",
                       bgcolor=th.GREEN, bordercolor=th.GREEN, borderwidth=0,
                       font=dict(color="#0B0F17", size=14, family="IBM Plex Mono, monospace"))
    th.plotly_layout(fig, height=320, margin=dict(l=40, r=12, t=40, b=30))
    fig.update_xaxes(title="predicted confidence")
    fig.update_yaxes(title="observed accuracy", range=[0, 1])
    return fig, float(ece)


def coverage_curve(probs, y, scores, labels, colors):
    """Interactive accuracy-vs-coverage curve for several uncertainty scores."""
    fig = go.Figure()
    grid = np.linspace(0.99, 0.05, 40)
    for i, (name, unc) in enumerate(scores.items()):
        ys, xs = [], []
        for cov in grid:
            n_keep = int(round(cov * len(probs)))
            if n_keep < 1:
                continue
            order = np.argsort(unc)[:n_keep]
            ys.append(float((probs[order].argmax(-1) == y[order]).mean()))
            xs.append(cov)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=labels.get(name, name),
                                 line=dict(color=colors[i % len(colors)], width=2.5)))
    th.plotly_layout(fig, height=360, margin=dict(l=40, r=12, t=46, b=30))
    fig.update_xaxes(title="coverage (fraction of inputs acted on)")
    fig.update_yaxes(title="accuracy on accepted inputs")
    return fig


def barometer(probs, pred, highlight=th.GREEN):
    """Horizontal per-class probability bars (instrument readout)."""
    order = np.argsort(probs)[::-1]
    labels = [th.CLASSES[i] for i in order]
    vals = probs[order]
    pos = int(np.where(order == pred)[0][0])
    text = [""] * len(order)
    text[pos] = f"{vals[pos]:.0%}"
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", name="probability",
        text=text, textposition="outside",
        marker_color=[highlight if i == pred else th.MUTED for i in order],
        marker_line=dict(width=0.5, color=th.PANEL2),
        hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    th.plotly_layout(fig, height=330, title="class barometer · top = most likely",
                     margin=dict(l=8, r=48, t=34, b=6))
    fig.update_xaxes(title="probability", range=[0, 1], tickformat=".0%",
                     showticklabels=True)
    fig.update_yaxes(tickfont=dict(size=11))
    return fig


def dist_histogram(values, threshold, title, xlabel):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=40, name="stream",
                               marker_color=th.VIOLET,
                               marker_line=dict(color=th.PANEL2, width=0.5),
                               opacity=0.9))
    if threshold is not None and np.isfinite(threshold):
        fig.add_vline(x=threshold, line_color=th.RED, line_width=2,
                      line_dash="dash")
        fig.add_annotation(x=threshold, yref="paper", y=0.98, ax=30, ay=-10,
                           text=f"abstain beyond {threshold:.3f}",
                           font=dict(color=th.RED, size=11))
    th.plotly_layout(fig, height=280, margin=dict(l=36, r=12, t=30, b=30))
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title="count")
    return fig


# ---------------------------------------------------------------------------
# cached analytics (fast path: pre-computed final results; slow path: live)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def final_results():
    p = os.path.join(config.RESULT_DIR, "final_eval.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_resource(show_spinner="Computing baseline clean-test stats…")
def clean_stats():
    res = engine.get_resources()
    from src.data import get_app_datasets
    from src.evaluate import score_dataset
    _, test_ds = get_app_datasets()
    loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)
    pb, yb, sb, _ = score_dataset(res["base"], loader, res["temp"], res["device"])
    mc_p, mc_y, mc_s, _ = score_dataset(res["base"], loader, res["temp"],
                                        res["device"], do_mc=True, n_mc=12)
    return dict(probs=pb, y=yb, scores=sb, mc_entropy=mc_s["mc_entropy"])


@st.cache_resource(show_spinner="Interrogating input…")
def _verdict(source, idx, corr, sev, coverage, upload_key):
    if source == "uploaded":
        return engine.predict_one(engine.upload_buffer[upload_key],
                                  coverage=coverage)
    if source == "clean":
        _, test_ds = engine_get_clean()
        x = test_ds[idx][0]
    else:
        xs, _ = engine.get_shift_data(corr, sev)
        x = xs[idx]
    return engine.predict_one(x, coverage=coverage)


@st.cache_resource(show_spinner="Running batch scan…")
def _scan(corr, sev, coverage, kind):
    if kind == "clean":
        return engine.scan_clean(coverage=coverage)
    return engine.scan_shift(corr, sev, coverage=coverage)


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
def page_mission_control():
    res = engine.get_resources()
    profile = res["profile"]
    state = ("ENSEMBLE · 4 MEMBERS · LIVE" if profile == "ensemble"
             else "MC-DROPOUT FALLBACK · ENSEMBLE TRAINING")
    st.markdown(th.sysbar("Mission Control", state), unsafe_allow_html=True)
    st.markdown(th.mission(
        '<p class="lead">A classifier reports <b>72% confidence</b> on a frame of '
        "pure noise — because neural networks can be <b>confidently wrong</b> when "
        "they meet inputs unlike anything they were trained on. This panel is the "
        "guardrail: every prediction is pushed through <b>calibrated confidence</b>, "
        "<b>ensemble disagreement</b> and <b>feature-space familiarity</b>, and — like "
        "a pilot&#39;s instruments — the system tells you the one thing that matters: "
        "<b>can I trust this call right now?</b></p>"), unsafe_allow_html=True)

    c1, c2 = st.columns([1.4, 2.6], gap="large")
    with c1:
        st.markdown('<div class="panel" style="margin-top:0;">'
                    '<h3>Input Feed</h3></div>', unsafe_allow_html=True)
        _, test_ds = engine_get_clean()
        source = st.radio("source", ["Clean CIFAR-10", "Corrupted sample",
                                     "Upload image"], horizontal=True)
        idx = st.slider("sample index", 0, 9999, 137)
        corr = "gaussian_noise"; sev = 5
        if source == "Corrupted sample":
            corr = st.selectbox("corruption", engine.shift_catalog())
            sev = st.selectbox("severity", engine.shift_severities(), index=4)
        coverage = st.slider("coverage target (act only on the most reliable)",
                             0.6, 1.0, 0.9, 0.05,
                             help="Fraction of inputs the system agrees to act on; "
                                  "the abstention threshold is learned from "
                                  "calibration data at exactly this coverage.")

        if source == "Upload image":
            up = st.file_uploader("32×32 RGB image", type=["png", "jpg", "jpeg"])
            if up is None:
                st.info("Upload an image to interrogate.")
                return
            img = Image.open(io.BytesIO(up.read())).convert("RGB").resize((32, 32))
            import hashlib
            upload = (np.asarray(img).astype(np.float32) / 255.0).transpose(2, 0, 1)
            upload = torch.from_numpy(upload)
            upload_key = hashlib.sha1(upload.numpy().tobytes()).hexdigest()
            engine.upload_buffer[upload_key] = upload

            v = _verdict("uploaded", idx, corr, sev, coverage, upload_key)
            x = upload
            origin = "uploaded image"
        else:
            src_kind = "clean" if source == "Clean CIFAR-10" else "corrupted"
            if source == "Clean CIFAR-10":
                _, test_ds = engine_get_clean()
                x = test_ds[idx][0]
                origin = f"clean test #{idx}"
            else:
                xs, _ = engine.get_shift_data(corr, sev)
                x = xs[idx]
                origin = f"{corr} sev{sev} #{idx}"
            v = _verdict(src_kind, idx, corr, sev, coverage, "")

        st.markdown('<div style="text-align:center;margin-top:10px;">'
                    '<span style="font-family:IBM Plex Mono,monospace;'
                    'color:#8B98AC;font-size:.7rem;">PIXEL FEED · 32×32 '
                    '</span></div>', unsafe_allow_html=True)
        st.image(x.permute(1, 2, 0).cpu().numpy(), width=250)
        vcol = {"TRUST": "pass", "REVIEW": "warn", "ABSTAIN": "fail"}[v["verdict"]]
        vlcol = {"TRUST": th.GREEN, "REVIEW": th.AMBER, "ABSTAIN": th.RED}[v["verdict"]]
        st.markdown(
            f'<div class="verdict {vcol}">{th.status_light(vlcol)}'
            f'&nbsp; SYSTEM {v["verdict"]}</div>', unsafe_allow_html=True)
        st.caption(f"input: {origin}")

    with c2:
        st.markdown('<div class="panel" style="margin-top:0;"><h3>Trust Dossier</h3></div>',
                    unsafe_allow_html=True)
        pred_label = th.CLASSES[v["pred"]]
        conf = v["scores"]["cal_conf"]
        r1, r2, r3 = st.columns(3)
        with r1:
            st.plotly_chart(th.gauge("confidence", conf, 1.0, th.GREEN),
                            width="stretch")
        with r2:
            fig = th.gauge("predictive entropy", v["scores"]["ens_entropy"],
                           max(1.2, v["scores"]["ens_entropy"] * 1.4), th.VIOLET)
            st.plotly_chart(fig, width="stretch")
        with r3:
            md = v.get("mahalanobis_dist", float("nan"))
            st.plotly_chart(th.gauge("familiarity score (Mahalanobis)", md,
                                     max(5.0, md * 1.3) if np.isfinite(md) else 5.0,
                                     th.VIOLET), width="stretch")

        checks_html = "".join(th.check_row(
            "ok" if ms == "pass" else ("bad" if ms.startswith("OOD") else "mid"),
            nm, ms) for nm, ms in v["checks"])
        st.markdown(f'<div class="panel"><h3>Guardrail Checks</h3>{checks_html}</div>',
                    unsafe_allow_html=True)

        st.markdown(f"""
        <div class="readout" style="margin-top:10px;">
          <div class="k">verdict rationale</div>
          <div class="v">{v.get("reason", " · ".join(f"{a}={b}" for a, b in v["checks"]))}</div>
        </div>
        <div class="readout" style="margin-top:8px;">
          <div class="k">system provenance</div>
          <div class="v">profile = {v["profile"]} · T = {res["temp"]:.3f} ·
          calibration coverage = {coverage:.2f} · members = {len(res["members"]) if res["members"] else "1 (+MC×12)"}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="panel" style="margin-top:14px;"><h3>Class Barometer</h3></div>',
                unsafe_allow_html=True)
    st.plotly_chart(barometer(v["probs"], v["pred"]), width="stretch")
    st.caption(f"Top prediction places the input in the **{pred_label}** class box; "
               "the dial to its left shows the calibrated degree of belief.")


@st.cache_resource(show_spinner=False)
def engine_get_clean():
    from src.data import get_app_datasets
    return get_app_datasets()


def page_drift_monitor():
    res = engine.get_resources()
    state = ("ENSEMBLE · 4 MEMBERS · LIVE" if res["profile"] == "ensemble"
             else "MC-DROPOUT FALLBACK · ENSEMBLE TRAINING")
    st.markdown(th.sysbar("Drift Monitor", state), unsafe_allow_html=True)
    st.markdown(th.mission(
        '<p class="lead">In production the input stream will drift: compression, '
        "sensor noise, fog. The model&#39;s accuracy collapses while its confidence "
        "stays high. This panel feeds a <b>corrupted stream</b> through the exact "
        "deployment pipeline and reports how much of it the system correctly "
        "<b>refuses to act on</b>.</p>"), unsafe_allow_html=True)

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        corr = st.selectbox("drift type", engine.shift_catalog())
    with c2:
        sev = st.selectbox("severity", engine.shift_severities(), index=4)
    coverage = st.slider("coverage target", 0.6, 1.0, 0.9, 0.05)

    scan = _scan(corr, sev, coverage, "shift")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("inputs scanned", f"{scan['n']}")
    m2.metric("model accepts", f"{scan['trust_rate']:.1%}",
              help="share of the stream the system acted on")
    m3.metric("accuracy (overall)", f"{scan['accuracy_overall']:.1%}" if np.isfinite(scan['accuracy_overall']) else "—")
    m4.metric("accuracy (accepted)", f"{scan['accuracy_on_trusted']:.1%}" if np.isfinite(scan['accuracy_on_trusted']) else "—")

    if (np.isfinite(scan["accuracy_on_trusted"])
            and scan["trust_rate"] > 0.7
            and scan["accuracy_on_trusted"] < 0.5):
        st.warning(
            "WATCHPOINT — the fallback guardrail is trusting almost everything "
            f"({scan['trust_rate']:.0%}) while being right only "
            f"{scan['accuracy_on_trusted']:.0%} of the time. Confidence and "
            "MC-Dropout alone cannot see this kind of damage; the deep-ensemble + "
            "feature-familiarity profile (thermal) flags it.")

    a, b = st.columns(2)
    with a:
        st.markdown('<div class="panel"><h3>Predictive entropy distribution</h3></div>',
                    unsafe_allow_html=True)
        st.plotly_chart(dist_histogram(
            scan["entropy"], scan["trust_threshold"],
            "entropy", "predictive entropy (higher = the system refuses)"),
            width="stretch")
    with b:
        st.markdown('<div class="panel"><h3>Feature-space familiarity</h3></div>',
                    unsafe_allow_html=True)
        if np.isfinite(scan["dist"]).all() and scan["dist"].size:
            thr = _maha_threshold(res, coverage)
            st.plotly_chart(dist_histogram(scan["dist"], thr,
                                           "mahalanobis", "min class distance"),
                            width="stretch")
        else:
            st.info("Feature-space OOD signal available once the deep ensemble "
                    "profile is live (requires the ensemble + fitted stats).")

    flagged = np.flatnonzero(scan["flagged"])
    if flagged.size:
        n = min(12, flagged.size)
        st.markdown('<div class="panel"><h3>Flagged inputs — the system refused these</h3></div>',
                    unsafe_allow_html=True)
        cols = st.columns(6)
        for i, fi in enumerate(flagged[np.random.RandomState(1).choice(len(flagged), n, replace=False)]):
            img = scan["x"][fi]
            prov = {}
            pred = th.CLASSES[scan["preds"][fi]]
            truth = th.CLASSES[scan["labels"][fi]] if np.isfinite(scan["labels"][fi]) else "?"
            with cols[i % 6]:
                st.image(img.permute(1, 2, 0).cpu().numpy(), width=96)
                st.caption(f"[{scan['verdicts'][fi][:3]}] pred {pred} / truth {truth}")
    else:
        st.success("No inputs were flagged at this coverage — the stream looks "
                   "compatible with the training distribution.")


def _maha_threshold(res, coverage):
    try:
        tc = engine._fit_trust_thresholds(coverage, res["device"], res["profile"])
        return tc.thresholds.get("mahalanobis", np.inf)
    except Exception:
        return np.inf


def page_calibration():
    res = engine.get_resources()
    state = ("ENSEMBLE · 4 MEMBERS · LIVE" if res["profile"] == "ensemble"
             else "MC-DROPOUT FALLBACK · ENSEMBLE TRAINING")
    st.markdown(th.sysbar("Calibration Lab", state), unsafe_allow_html=True)
    st.markdown(th.mission(
        '<p class="lead">Confidence is only useful if it is <b>honest</b>: a model '
        "that says 80% should be right 80% of the time. The lab measures this "
        "with the <b>Expected Calibration Error</b> before and after "
        "<b>temperature scaling</b>, and shows what we give up in accuracy for "
        "reliability.</p>"), unsafe_allow_html=True)

    sts = clean_stats()
    probs, y = sts["probs"], sts["y"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel" style="margin-top:0;"><h3>Raw softmax (T = 1.0)</h3></div>',
                    unsafe_allow_html=True)
        fig_raw, ece_raw = reliability_plot(probs, y, "raw")
        st.plotly_chart(fig_raw, width="stretch")
    with c2:
        st.markdown(f'<div class="panel" style="margin-top:0;"><h3>Temperature-scaled '
                    f'(T = {res["temp"]:.3f})</h3></div>', unsafe_allow_html=True)
        pb2 = np.power(probs, 1 / res["temp"]) / np.sum(np.power(probs, 1 / res["temp"]),
                                                        axis=-1, keepdims=True)
        fig_s, ece_s = reliability_plot(pb2, y, "scaled")
        st.plotly_chart(fig_s, width="stretch")

    a, b, c, d = st.columns(4)
    a.metric("ECE raw", f"{ece_raw:.4f}", delta=f"{-ece_raw:.4f}")
    b.metric("ECE scaled", f"{ece_s:.4f}", delta=f"{ece_s - ece_raw:+.4f}" if ece_s < ece_raw else f"{ece_s - ece_raw:+.4f}")
    c.metric("base accuracy", f"{float((probs.argmax(-1) == y).mean()):.1%}")
    d.metric("Brier (raw)", f"{_brier(probs, y):.4f}")

    st.markdown('<div class="panel"><h3>Selective prediction — accuracy as we restrict coverage</h3></div>',
                unsafe_allow_html=True)
    scores = {"softmax": sts["scores"]["softmax"],
              "entropy": sts["scores"]["entropy"],
              "mc_entropy": sts["mc_entropy"]}
    labels = {"softmax": "softmax (MSP)", "entropy": "predictive entropy",
              "mc_entropy": "ens. entropy (MC)"}
    colors = [th.SKY, th.AMBER, th.VIOLET]
    st.plotly_chart(coverage_curve(probs, y, scores, labels, colors),
                    width="stretch")
    st.caption("Every method abstains on its own most-uncertain predictions. The "
               "accuracy of the remaining set climbs as coverage shrinks.")


def _brier(probs, y):
    onehot = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=-1)))


def page_showdown():
    res = engine.get_resources()
    st.markdown(th.sysbar("Method Showdown",
                          "ENSEMBLE · 4 MEMBERS · LIVE" if res["profile"] == "ensemble"
                          else "MC-DROPOUT FALLBACK · ENSEMBLE TRAINING"),
                unsafe_allow_html=True)
    st.markdown(th.mission(
        '<p class="lead">Four families of uncertainty were built and measured: '
        "<b>softmax confidence</b>, <b>MC-Dropout</b>, a <b>deep ensemble</b> and "
        "a <b>Mahalanobis feature-distance</b> detector. Which one can you bet a "
        "deployment on? The table speaks — and the ensemble dominates on nearly "
        "every axis.</p>"), unsafe_allow_html=True)

    fr = final_results()
    if fr is None:
        st.info("Full method comparison requires the final evaluation pass "
                "(`python scripts/05_final_evaluation.py`). Showing the live "
                "baseline subset meanwhile.")
        sts = clean_stats()
        probs, y = sts["probs"], sts["y"]
        scores = {"softmax": sts["scores"]["softmax"],
                  "entropy": sts["scores"]["entropy"],
                  "mc_entropy": sts["mc_entropy"]}
        mask_c = (probs.argmax(-1) == y)
        rows = []
        from src import metrics as mt
        for name, unc in scores.items():
            rows.append({"method": name,
                         "error-AUROC": float(mt.error_detection_auroc(unc, mask_c))})
        a, b = st.columns(2)
        with a:
            _plot_auroc_bars(rows)
        with b:
            st.plotly_chart(coverage_curve(
                probs, y, scores,
                {"softmax": "softmax (MSP)", "entropy": "entropy",
                 "mc_entropy": "MC entropy"},
                [th.SKY, th.AMBER, th.VIOLET]), width="stretch")
        return

    f = fr["clean"]
    rows = []
    lab = fr["method_labels"]
    for name in f["error_auroc"]:
        rows.append({"method": lab.get(name, name), "error-AUROC": f["error_auroc"][name]})
    rows.sort(key=lambda r: r["error-AUROC"], reverse=True)

    a, b = st.columns(2)
    with a:
        st.markdown('<div class="panel" style="margin-top:0;"><h3>Error-detection AUROC — clean test</h3></div>',
                    unsafe_allow_html=True)
        _plot_auroc_bars(rows)
    with b:
        st.markdown('<div class="panel" style="margin-top:0;"><h3>IN vs OUT separation (shift)</h3></div>',
                    unsafe_allow_html=True)
        tbl = fr["auroc_table"]
        methods = sorted({k.split("@")[0] for k in tbl})
        outs = sorted({k.split("@")[1] for k in tbl})
        fig = go.Figure()
        palette = dict(zip(methods, [th.SKY, th.AMBER, th.VIOLET, th.GREEN,
                                     th.RED, "#F472B6", "#22D3EE"]))
        for o in outs:
            fig.add_trace(go.Bar(x=methods, y=[tbl[f"{m}@{o}"] for m in methods],
                                 name=o, marker_color=palette.get(o, th.MUTED)))
        th.plotly_layout(fig, height=330, margin=dict(l=40, r=12, t=40, b=60))
        fig.update_yaxes(title="AUROC", range=[0, 1])
        st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="panel"><h3>Coverage behaviour — clean test (accuracy at coverage)</h3></div>',
                unsafe_allow_html=True)
    cov = f["coverage"]
    methods_c = [lab.get(k, k) for k in cov]
    covs = ["cov_100", "cov_95", "cov_90", "cov_80", "cov_70", "cov_50"]
    names = [f"{c.removeprefix('cov_')}%" for c in covs]
    fig = go.Figure()
    for k in cov:
        fig.add_trace(go.Scatter(x=names, y=[cov[k][c] for c in covs],
                                 mode="lines+markers", name=lab.get(k, k),
                                 line=dict(width=2.5)))
    th.plotly_layout(fig, height=360, margin=dict(l=40, r=12, t=46, b=30))
    fig.update_yaxes(title="accuracy")
    st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="panel"><h3>Abstention payoff under distribution shift</h3></div>',
                unsafe_allow_html=True)
    rows = []
    for env, methods in fr["abstain"].items():
        for m, v in methods.items():
            rows.append({"environment": env, "method": lab.get(m, m),
                         "accuracy@90%": v["acc_at_90"],
                         "gain": v["improvement"]})
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)

    st.markdown('<div class="panel"><h3>Deployed verdict (deep ensemble + Mahalanobis)</h3></div>',
                unsafe_allow_html=True)
    ver = fr.get("verdicts", {})
    env_vert = []
    for env, v in ver.items():
        env_vert.append({"environment": env,
                         "fraction trusted": f"{v['fraction_trusted']:.1%}",
                         "accuracy all": f"{v['acc_all']:.1%}",
                         "accuracy on trusted": (f"{v['acc_on_trusted']:.1%}"
                                                 if np.isfinite(v.get("acc_on_trusted", float('nan'))) else "—")})
    if env_vert:
        st.dataframe(env_vert, width="stretch", hide_index=True)
    st.caption("The deep ensemble + familiarity guardrail dramatically raises "
               "accuracy on accepted inputs, at an explicit coverage cost — "
               "the trade created by teaching the system *when not to trust* its "
               "own prediction.")


def _plot_auroc_bars(rows):
    mm = [r["method"] for r in rows]
    vals = [r["error-AUROC"] for r in rows]
    colors = [th.GREEN if v == max(vals) else th.MUTED for v in vals]
    fig = go.Figure(go.Bar(x=mm, y=vals, name="error-AUROC", marker_color=colors,
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    th.plotly_layout(fig, height=330, margin=dict(l=40, r=12, t=40, b=60))
    fig.update_yaxes(title="AUROC", range=[0, 1])
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------
def main():
    with st.sidebar:
        st.markdown('<div style="font-size:1.1rem;font-weight:700;letter-spacing:.02em;">'
                    '🛡️ TrustGuard</div>', unsafe_allow_html=True)
        page = st.radio("panel", PAGES, label_visibility="collapsed")
        st.markdown("---")
        res = engine.get_resources()
        prof = res["profile"]
        st.markdown(f"""
        <div class="readout">
          <div class="k">pipeline status</div>
          <div class="v">{'🟢 ensemble' if prof == 'ensemble' else '🟠 mc-fallback'}</div>
          <div class="k" style="margin-top:6px;">temperature T</div>
          <div class="v">{res['temp']:.3f}</div>
          <div class="k" style="margin-top:6px;">device</div>
          <div class="v">{res['device'].upper()}</div>
        </div>""", unsafe_allow_html=True)
        st.caption("Build · Learn.I-learn · Reliability Stack (A/B/C/D + fusion)")

    pages = {"Mission Control": page_mission_control,
             "Drift Monitor": page_drift_monitor,
             "Calibration Lab": page_calibration,
             "Method Showdown": page_showdown}
    pages[page]()


if __name__ == "__main__":
    main()