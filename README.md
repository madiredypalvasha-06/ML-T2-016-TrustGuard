# TrustGuard — Teaching a Machine Learning System When *Not* to Trust Its Own Prediction

**Project ID:** ML-T2-016 · **Track 2 (Advanced ML) Internship** — Learn Depth™ Academy LLP
**Author:** Palvasha Madireddy · **Domain:** ML Reliability · Uncertainty Estimation · Selective Classification

> A classifier that says **"I don't know"** — and can prove it.

Standard image classifiers are evaluated only on accuracy over in-distribution data, so
they are judged on a metric that says nothing about their behaviour *outside* that data.
In deployment the input stream drifts, and the network does not degrade gracefully — it
becomes **confidently wrong**. A model that reports 95% confidence when its true
accuracy on that slice is 15% is not merely inaccurate; it is *undetectably* inaccurate.

This project builds a **trust-aware image classifier** that emits a class prediction,
a calibrated reliability signal, **and** an explicit, auditable
`TRUST / REVIEW / ABSTAIN` verdict that a human operator can inspect and override.

---

## The headline finding

The same trained model, the same checkpoint, two environments:

| Environment | Accuracy | Mean confidence | ECE | Softmax IN-vs-OUT AUROC |
|---|---|---|---|---|
| **Clean held-out test (10k)** | **72.02%** | 0.60 | **0.0406** (well calibrated) | — |
| **Same model + severe Gaussian noise** | **16.38%** | **0.84** ⬆️ | **0.6774** ⬆️ | **0.3082** ⬇️ |

Confidence goes **up** while accuracy collapses. And the softmax signal's
out-of-distribution AUROC is **0.31 — worse than a coin flip**. It points the *wrong
way*. This is the cleanest empirical demonstration that a standard confidence score
does not tell you when the model should not be trusted.

Which method *does* catch those inputs?

| Method | IN-vs-OUT AUROC (Gaussian noise, sev 5) | Verdict |
|---|---|---|
| A. Softmax / MSP | 0.3082 | inverted |
| D. Predictive entropy | 0.2904 | inverted |
| B. MC-Dropout entropy | 0.3132 | inverted |
| C. Deep-ensemble entropy | 0.5512 | weak |
| C. Deep-ensemble disagreement | 0.7203 | decent |
| C. Deep-ensemble **variance** | **0.9021** | ✅ |
| D. **Mahalanobis** familiarity | **0.9187** | ✅ |

**Conclusion:** *answer certainty* and *input familiarity* are different quantities, and
you need both. Confidence is a good **in-distribution error detector** (AUROC 0.8178)
and a **bad out-of-distribution detector**. This complementarity is the empirical
justification for the fused design in `TrustAwareClassifier`.

---

## What the system does

* A **BatchNorm-free 7-layer CNN** (exactly 2,196,810 parameters) classifies CIFAR-10.
* **Seven uncertainty signals** across four method families are implemented and
  benchmarked head-to-head on identical data:

  | Family | Method | Signals |
  |---|---|---|
  | **A** | Softmax / Maximum Softmax Probability (Hendrycks & Gimpel 2017) | `softmax` (1 − max p) |
  | **A2** | Temperature scaling (Guo et al. 2017) | fitted `T = 1.1824` |
  | **B** | Monte-Carlo Dropout, K = 30 (Gal & Ghahramani 2016) | predictive entropy, variance, mutual information |
  | **C** | Deep ensemble, M = 4 (Lakshminarayanan et al. 2017) | ensemble entropy, variance, mutual information, fraction-disagreeing |
  | **D** | Mahalanobis feature familiarity (Lee et al. 2018) | min-class Mahalanobis distance, Ledoit–Wolf pooled covariance |

* Evaluation goes beyond accuracy, along **four reliability axes**:
  calibration (ECE + reliability diagrams), **risk–coverage**, **error-detection
  AUROC**, and **IN-vs-OUT separation AUROC**.
* **`TrustAwareClassifier`** (`src/ensembles.py`) fuses calibrated confidence +
  ensemble entropy + feature familiarity into a single three-way verdict with
  human-readable per-check reasons:
  `confidence=LOW · uncertainty=HIGH · familiarity=OOD(d=14.2)`
* **TrustGuard** — a Streamlit "mission control" dashboard with four instrument
  panels (Mission Control, Drift Monitor, Calibration Lab, Method Showdown) makes
  every one of these claims interactively inspectable.

## More results

**Selective prediction on clean data** — acting only on the most reliable inputs:

| Method | 100% cov | 90% cov | 70% cov | 50% cov |
|---|---|---|---|---|
| A. Softmax (MSP), single model | 0.7202 | 0.7629 | 0.8424 | **0.9162** |
| D. Entropy, single model | 0.7202 | 0.7570 | 0.8314 | 0.9050 |
| C. Deep ensemble (entropy) | 0.7718 | 0.8122 | 0.8791 | **0.9336** |
| C. Deep ensemble (disagreement) | 0.7718 | 0.7973 | 0.8676 | 0.9068 |
| D. Mahalanobis | 0.7718 | 0.7567 | 0.7279 | 0.7022 ⚠️ |

Abstaining on the least-reliable half of clean inputs lifts single-model accuracy
**72.02% → 91.62%** (+19.6 pp) with no retraining.

⚠️ **Worth noting:** Mahalanobis is the *only* signal whose risk–coverage curve slopes
**downward**. Ranking by "how familiar is this input" and abstaining on the most
unfamiliar ones makes accuracy *worse* (0.77 → 0.70) — because on clean in-distribution
data, unfamiliarity is not a reliable proxy for error. Familiarity earns its place in
the **shift** experiments (§6.5), not the clean ones. A signal that helps in one
regime and hurts in another is exactly why a single-score system is the wrong design.

**The deployed fused verdict** (thresholds fit at 90% calibration coverage):

| Environment | Fraction trusted | Accuracy (all) | Accuracy (trusted only) |
|---|---|---|---|
| clean | 0.800 | 0.7718 | **0.7965** |
| fog sev 5 | 0.838 | 0.7312 | **0.7719** |
| Gaussian noise sev 5 | **0.223** | 0.1682 | 0.1244 |

The system trusts 80% of clean data and is *more* accurate on what it accepts than on
the population (79.65% vs 77.18%). Under severe noise it **refuses ~78%** of the
stream — the behaviour a naive classifier cannot produce. The 0.1244 is reported
honestly: when corruption destroys the signal in *every* image, per-input filtering
cannot manufacture reliability, and only the aggregate trust rate is informative. See
§6.7 of the technical paper.

**Degradation under shift** (15 conditions, 5 corruptions × 3 severities) —
accuracy, ECE and mean confidence for each are tabulated in §6.4 of the paper. The
model is resilient to *fog* (72% accuracy at severity 5) and catastrophically fragile
to *brightness* and *high-severity Gaussian noise* (11.5% and 16.4%).

## Known design choices (documented for reproducibility)

* **Shift is generated locally.** The standard CIFAR-10-C corruption recipes
  (gaussian noise, motion blur, fog, brightness, contrast) are re-implemented
  deterministically in `src/corruptions.py` with the published severity parameters.
  The original CIFAR-10-C GitHub release returns an empty archive, so generating the
  corruptions locally makes the study self-contained and reproducible with no
  external download.
* **No BatchNorm anywhere.** `model.train()` at inference must enable *only* dropout,
  otherwise MC-Dropout is not a well-defined stochastic estimator. This is a
  deliberate correctness-over-convenience choice and we accept the small clean-accuracy
  cost (see §7.2 of the paper).
* **Disciplined splits.** 45,000 train / 5,000 calibration / 10,000 held-out test. The
  calibration split is used *only* to fit temperature, the Mahalanobis statistics and
  the abstention thresholds. The test set is touched exactly once, at final evaluation.
* **Every metric is hand-implemented** in `src/metrics.py` (ECE, Brier, risk–coverage,
  rank-based AUROC, error-detection AUROC) rather than pulled from a library, so the
  definitions are explicit and auditable.

## Project layout

```
app/
  app.py              TrustGuard dashboard (4 instrument panels)
  engine.py           model / trust-profile loading + batch drift scanners
  theme.py            instrument-panel design system (CSS + Plotly)
  streamlit_app.py    lightweight single-input MC-Dropout demo
src/
  config.py           paths, hyper-parameters, seeds, device selection
  data.py             CIFAR-10 splits + shifted-tensor generation
  corruptions.py      deterministic CIFAR-10-C-style corruptions
  models.py           CIFAR10CNN (dropout-enabled, exposes penultimate features)
  uncertainty.py      temperature scaling, MC dropout, entropy scores
  ensembles.py        deep ensemble + TrustAwareClassifier (fused verdict)
  mahalanobis.py      feature-space OOD distance (Ledoit-Wolf pooled covariance)
  metrics.py          ECE, Brier, risk-coverage, AUROC, reliability plots
  evaluate.py         shared evaluation pipeline
scripts/
  01_train_baseline.py     train a CNN (--seed / --tag / --epochs / --cpu / --init)
  02_evaluate_clean.py     calibration + clean-test evaluation
  03_shift_experiments.py  shift / OOD / risk-coverage experiments
  04_report.py             regenerate results/report.md from the JSON artefacts
  05_final_evaluation.py   ensemble + Mahalanobis + fused-verdict evaluation
  06_watch_and_finish.sh   poll for ensemble checkpoints, then run steps 5 and 4
checkpoints/     trained weights, fitted temperature, Mahalanobis statistics
results/         JSON metrics, figures/ (12 PNGs), auto-generated report.md
Technical_Paper_ML-T2-016.md   full technical report (10 sections, 12 references)
Stage1_Research_Brief_ML-T2-016.md   Stage-1 research brief
Development.py   pipeline driver (all | train | evaluate | shift | final | report | demo)
```

## Quick start

```bash
pip install -r requirements.txt

# 1. train the baseline CNN (CIFAR-10 auto-downloads into data/ on first run)
python scripts/01_train_baseline.py

# 2. clean-set calibration + uncertainty evaluation
python scripts/02_evaluate_clean.py

# 3. distribution-shift experiments
python scripts/03_shift_experiments.py

# 4a. train the remaining deep-ensemble members (~16 h each on CPU; run in parallel)
python scripts/01_train_baseline.py --seed 99   --tag member_1 --epochs 20 --cpu &
python scripts/01_train_baseline.py --seed 55   --tag member_2 --epochs 20 --cpu \
       --init checkpoints/baseline_cnn.pt &
python scripts/01_train_baseline.py --seed 1234 --tag member_3 --epochs 20 --cpu &
cp checkpoints/baseline_cnn.pt checkpoints/member_0.pt

# 4b. final evaluation (ensemble + Mahalanobis + fused verdict) -> results/final_eval.json
python scripts/05_final_evaluation.py

# 5. rebuild the markdown report
python scripts/04_report.py

# 6. launch the TrustGuard dashboard
python Development.py demo          # -> http://localhost:8501
```

Or drive the whole pipeline: `python Development.py all`.

Everything runs on a CPU laptop. On Apple Silicon, inference is dispatched to the MPS
backend automatically via `config.get_device()`; training stays on CPU because it was
more stable in this environment.

> **Runtime note, stated honestly:** the three ensemble members take ~16–17 hours each
> on a laptop CPU. Steps 2–3, 4b, 5 and the dashboard all complete in minutes. A
> reviewer who only wants to inspect the results can read `results/final_eval.json` and
> launch the dashboard directly — steps 1 and 4a are only needed to regenerate
> checkpoints from scratch.

## Evaluation metrics used

| Metric | What it answers |
|---|---|
| **ECE** (15 bins) | How far is stated confidence from observed accuracy? |
| **Brier score** | Proper scoring rule for the full probability vector |
| **Risk–coverage** | What accuracy do we get on the predictions the model *keeps*? |
| **Error-detection AUROC** | Does the uncertainty score rank wrong predictions above right ones? |
| **IN-vs-OUT AUROC** | Does the score separate clean from corrupted inputs? |
| **Trust rate / accepted accuracy** | At deployment, how much do we accept and how right is it? |

## Reproducibility

`SEED = 42` (`src/config.py`) drives the train/calibration split permutation; the
corruption kernels are seeded per call, so a given `(corruption, severity)` pair
always yields the same images. All fitted state — 4 checkpoints, `temp_scale.json`,
`mahalanobis_stats.npz` — is committed under `checkpoints/`, and all metrics are
committed under `results/`, so every number and figure in the technical paper can be
verified without retraining.

## References (methods implemented here)

1. Guo et al. (2017) — *On Calibration of Modern Neural Networks*, ICML.
2. Gal & Ghahramani (2016) — *Dropout as a Bayesian Approximation*, ICML.
3. Lakshminarayanan et al. (2017) — *Simple and Scalable Predictive Uncertainty using Deep Ensembles*, NeurIPS.
4. Hendrycks & Gimpel (2017) — *A Baseline for Detecting Misclassified and Out-of-Distribution Examples*, ICLR.
5. Lee et al. (2018) — *A Unified Approach to Interpreting Model Predictions*, ICML. (Mahalanobis attribution)
6. Hendrycks & Dietterich (2019) — *Benchmarking Neural Network Robustness to Common Corruptions*, ICLR. (CIFAR-10-C recipes)
7. El-Yaniv & Wiener (2010) — *On the Foundations of Noise-free Selective Classification*, JMLR.
8. Geifman & El-Yaniv (2017) — *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*, ICML.
9. Ovadia et al. (2019) — *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift*, NeurIPS.
10. Ledoit & Wolf (2004) — *A Well-Conditioned Estimator for Large-Dimensional Covariance*, JMLR.
11. Krizhevsky (2009) — *Learning Multiple Layers of Features from Tiny Images*, tech report. (CIFAR-10)
12. Scikit-learn — `sklearn.covariance.LedoitWolf` documentation.

## License

MIT — see [`LICENSE`](LICENSE).
