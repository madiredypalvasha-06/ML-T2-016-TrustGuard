# Stage 1 — Research Brief
**Track 2 · Advanced Machine Learning Internship — Learn Depth Academy LLP**
**Problem ID:** ML-T2-016
**Title:** Teaching a Machine Learning System When Not to Trust Its Own Prediction
**Domain:** ML Reliability · Uncertainty Estimation

---

## 1. Problem Understanding

Machine learning systems are normally judged by accuracy on data that resembles their training set. In real deployment, the system will inevitably see inputs that differ from what it was trained on — new conditions, edge cases, or data from a different source. A model can still output a confident, seemingly authoritative prediction in these situations even when that prediction is unreliable, because standard classifiers are not designed to know the limits of their own knowledge.

In plain terms: the problem is not "can the model predict correctly" but "can the model tell you when it *shouldn't* be trusted." This matters anywhere a wrong-but-confident prediction is costly — medical triage, fraud flags, safety systems — because a silent failure is far more dangerous than a system that says "I'm not sure."

## 2. Problem Formulation

| Element | Definition |
|---|---|
| **Problem statement** | Build a system that not only predicts, but also produces a reliability/uncertainty signal that reflects how trustworthy that specific prediction actually is. |
| **ML task** | Classification (base task) + uncertainty estimation / selective prediction (secondary task) |
| **Input** | Standard feature vector or image, same as any classifier — plus, during evaluation, deliberately shifted/out-of-distribution inputs |
| **Output / target** | (a) Class prediction, (b) a confidence/uncertainty score, (c) a binary "trust / abstain" decision at a chosen operating threshold |
| **Unit of prediction** | One prediction + one associated confidence score per input instance |
| **Assumptions** | Training data is representative of "normal" conditions; distribution shift can be simulated in a controlled way; ground-truth labels exist for evaluation |
| **Constraints** | Zero-cost, free/open-source tools only; must run on a laptop; 2–3 week timeline |
| **Success criteria** | Confidence scores correlate with actual correctness (calibration); accuracy improves when the system is allowed to abstain on low-confidence cases (risk–coverage trade-off); the system detects shifted/unfamiliar inputs better than raw softmax probability |

## 3. Use Case

**Who benefits:** Any team deploying an automated classifier in a setting where a wrong, over-confident answer is costly — e.g., a content moderation system, a medical screening tool, a fraud-detection pipeline, or a support-ticket triage system. This mirrors the "Designing an ML System That Knows When Not to Make the Decision" family of problems seen elsewhere in this problem set, but ML-T2-016 focuses specifically on the *uncertainty signal itself* rather than the downstream routing decision.

**Why it matters:** Deployed models silently degrade when their operating environment drifts. A system that can say "this input looks unfamiliar, treat this prediction cautiously" gives operators the chance to intervene before a bad decision compounds.

## 4. Initial Data Strategy

| Question | Investigation |
|---|---|
| Source | Public, well-known datasets — CIFAR-10 / CIFAR-10-C (corrupted version) for vision, or UCI Adult Income / Titanic for tabular. Both have permissive licenses and no cost. |
| Record unit | One image or one row = one prediction instance |
| Features | For image data: pixel data (I will use a small CNN). For tabular: standard demographic/behavioral columns. |
| Scale | CIFAR-10: 60,000 images (10 classes) — more than sufficient. Adult Income: ~48,000 rows — sufficient for a solid baseline + shift experiment. |
| Quality concerns | Corrupted-CIFAR variants already provide known perturbation types (blur, noise, fog), useful as a controlled OOD proxy. For tabular data, I'll manually engineer a distribution shift (e.g., train on one demographic slice, test on another). |
| Leakage risk | Must ensure the "shifted" test set is never used during training or hyperparameter tuning — kept strictly held out. |
| Legal use | Both datasets are open-source / academic-license, free for research and coursework use. |
| Preparation needed | Standard normalization/scaling; for CIFAR-10-C, no extra prep needed since it's pre-built; for tabular, I will artificially split into "in-distribution" and "shifted" subsets by a defensible rule (e.g., a demographic or geographic slice withheld from training). |

## 5. Existing Solutions

Relevant prior approaches investigated:
- **Softmax confidence / Maximum Softmax Probability (MSP)** — the simplest baseline; known to be poorly calibrated and overconfident, especially under shift.
- **Temperature scaling** (Guo et al., 2017) — a lightweight post-hoc calibration method that rescales logits to better match true accuracy.
- **Monte Carlo Dropout** (Gal & Ghahramani, 2016) — approximates Bayesian uncertainty by running multiple stochastic forward passes and measuring prediction variance.
- **Deep Ensembles** (Lakshminarayanan et al., 2017) — training multiple models and using their disagreement as an uncertainty signal; more expensive but often more reliable.
- **Out-of-distribution detection scores** (e.g., entropy of the softmax output, Mahalanobis distance in feature space) — used to flag inputs that look unlike training data.

## 6. Literature / Technical Research

| Source | Key takeaway | Relevance |
|---|---|---|
| Guo et al., "On Calibration of Modern Neural Networks" (2017) | Modern deep nets are systematically overconfident; temperature scaling is a cheap, effective fix | Direct baseline method for this project |
| Gal & Ghahramani, "Dropout as a Bayesian Approximation" (2016) | Dropout at inference time gives a usable uncertainty estimate without retraining | Candidate method B |
| Hendrycks & Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples" (2017) | Simple softmax-based OOD detection baseline | Useful as the naive baseline to beat |
| Hendrycks et al., "Benchmarking Neural Network Robustness to Common Corruptions" (2019) | Introduced CIFAR-10-C — a standard, ready-made corruption benchmark | Directly usable dataset for controlled shift experiments |
| El-Yaniv & Wiener, "On the Foundations of Noise-free Selective Classification" (2010) | Formalizes the risk–coverage trade-off, i.e., how accuracy changes as you abstain more | Provides the evaluation framework for this project |

## 7. Candidate Approaches

| Approach | Why it may work | Potential limitation |
|---|---|---|
| **A. Softmax probability + temperature scaling** | Cheap, well-documented, minimal extra compute, strong published baseline | Still limited under strong distribution shift — designed mainly to fix miscalibration, not detect novelty |
| **B. Monte Carlo Dropout** | Captures model uncertainty (not just data uncertainty), works with any dropout-enabled network, no retraining needed | Requires multiple forward passes at inference (slower); quality depends on dropout rate tuning |
| **C. Small Deep Ensemble (3–5 models)** | Generally the strongest uncertainty signal in the literature; disagreement is intuitive and easy to explain | More training time and memory; possibly tight for a 2–3 week zero-cost timeline |
| **D. Entropy/Mahalanobis-based OOD scoring** | Directly targets "is this input unfamiliar," complements confidence calibration | Needs careful feature-space setup; more implementation work than A or B |

**Initial direction:** Start with A (softmax + temperature scaling) as the baseline, then add B (MC Dropout) as the primary uncertainty method, and use D (entropy-based OOD score) as a lightweight complementary signal. C is a stretch goal if time permits.

## 8. Evaluation Strategy

- **Calibration:** Reliability diagrams + Expected Calibration Error (ECE) — do stated confidences match observed accuracy?
- **Risk–coverage curves:** As the system is allowed to abstain on an increasing fraction of low-confidence inputs, does accuracy on the remaining ("covered") predictions improve? This is the core "strong result" metric for this problem.
- **OOD separation:** Can the uncertainty score reliably distinguish in-distribution vs. corrupted/shifted inputs (measured via AUROC of uncertainty score vs. in/out-of-distribution label)?
- **Comparison across methods:** Baseline (raw softmax) vs. temperature scaling vs. MC Dropout vs. entropy score, evaluated on the same held-out shifted set.

## 9. Initial Methodology

| Area | Initial Plan |
|---|---|
| Data | CIFAR-10 (train/in-distribution) + CIFAR-10-C (shifted/OOD test set); tabular fallback = UCI Adult Income with an engineered shift split |
| Baseline | Small CNN (or logistic regression for tabular) trained normally; confidence = raw softmax max probability |
| Candidate models | Same base model + (1) temperature scaling, (2) MC Dropout at inference, (3) entropy-based OOD scoring |
| Evaluation | ECE, reliability diagrams, risk-coverage curves, AUROC for OOD detection |
| Experiments | (1) In-distribution calibration test, (2) performance under increasing corruption severity, (3) risk-coverage comparison across methods |
| Engineering | Simple Streamlit or CLI demo: upload/select an image → see prediction + confidence + "trust/flag for review" decision |
| Open questions | Which corruption severities best simulate realistic deployment drift? Is MC Dropout sufficient, or is a small ensemble worth the extra time? |

## 10. Open Questions

- Should the "strong result" bar be framed around vision (CIFAR-10-C) or tabular (Adult Income) — vision has better tooling (CIFAR-10-C is a ready benchmark), but tabular may be faster to iterate on. *Leaning toward vision for the ready-made shift benchmark.*
- How many severity levels of corruption should be tested to keep this within a 2–3 week scope without diluting the analysis?
- Is a lightweight deep ensemble (3 small models) feasible within the compute/time budget, or should it be dropped as a stretch goal only?
- What abstention threshold is "operationally reasonable" to present as a recommendation, and how should that be justified rather than picked arbitrarily?

## 11. References

1. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML.
2. Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML.
3. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. NeurIPS.
4. Hendrycks, D., & Gimpel, K. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks*. ICLR.
5. Hendrycks, D., & Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. ICLR. (Source of CIFAR-10-C dataset.)
6. El-Yaniv, R., & Wiener, Y. (2010). *On the Foundations of Noise-free Selective Classification*. JMLR.
7. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. (CIFAR-10 dataset technical report.)

---

*Prepared for Stage 1 submission — Learn Depth Academy LLP, Track 2 Advanced ML Internship, Project 01.*
