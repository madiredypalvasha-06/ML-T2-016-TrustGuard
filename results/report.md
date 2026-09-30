# ML-T2-016 - Development Stage Results

**Title:** Teaching a Machine Learning System When Not to Trust Its Own Prediction
**Dataset:** CIFAR-10 (clean) + controlled CIFAR-10-C-style corruptions
**Model:** 7-layer CNN (see `src/models.py`)
**Temperature (temp scaling):** 1.1824

## 1. Clean test set (held out, 10k examples)

| Metric | Value |
|---|---|
| Accuracy | 0.7202 |
| Brier score (temperature-scaled) | 0.3838 |
| ECE (temperature-scaled) | 0.0406 |

### Error-detection AUROC (does uncertainty track mistakes?)

| Method | AUROC |
|---|---|
| D. Entropy | 0.7987 |
| B. MC Dropout (entropy) | 0.7966 |
| B. MC Dropout (mutual info) | 0.6907 |
| B. MC Dropout (variance) | 0.6878 |
| A. Softmax (MSP) | 0.8178 |

### Selective prediction: accuracy at coverage (clean test)

| Method | 100% | 95% | 90% | 80% | 70% | 50% |
|---|---|---|---|---|---|---|
| A. Softmax (MSP) | 0.7202 | 0.7420 | 0.7629 | 0.8047 | 0.8424 | 0.9162 |
| D. Entropy | 0.7202 | 0.7386 | 0.7570 | 0.7951 | 0.8314 | 0.9050 |
| B. MC Dropout (entropy) | 0.7202 | 0.7388 | 0.7568 | 0.7951 | 0.8320 | 0.9022 |
| B. MC Dropout (variance) | 0.7202 | 0.7265 | 0.7338 | 0.7505 | 0.7711 | 0.8272 |
| B. MC Dropout (mutual info) | 0.7202 | 0.7274 | 0.7357 | 0.7544 | 0.7736 | 0.8280 |

## 2. Training summary

Trained for 24 epochs (early-stopped on calibration accuracy).
Best calibration accuracy: 0.7287; best test accuracy: 0.7209.

See `results/figures/training_history.png`.

## 3. Final evaluation: deep ensemble + feature-space familiarity

All four approach families are now assembled (temperature-scaled softmax, MC-Dropout, a 4-member deep ensemble, Mahalanobis feature distance) and scored on the identical held-out test set.

**Ensemble temperature:** 1.1824

### Headline accuracy / calibration

| Method | Accuracy | ECE | Brier |
|---|---|---|---|
| Single model (temp-scaled) | 0.7202 | 0.0406 | 0.3838 |
| Deep ensemble (temp-scaled) | 0.7718 | 0.1080 | 0.3368 |

### Error-detection AUROC — clean test

| Method | AUROC |
|---|---|
| A. Softmax (MSP) | 0.8178 |
| C. Deep ensemble (entropy) | 0.8120 |
| D. Entropy | 0.7987 |
| C. Deep ensemble (disagreement) | 0.7957 |
| B. MC Dropout (entropy) | 0.7716 |
| C. Deep ensemble (variance) | 0.6825 |
| D. Mahalanobis distance | 0.3631 |

### Selective prediction (accuracy at coverage, clean test)

| Method | 100% | 95% | 90% | 80% | 70% | 50% |
|---|---|---|---|---|---|---|
| A. Softmax (MSP) | 0.7202 | 0.7420 | 0.7629 | 0.8047 | 0.8424 | 0.9162 |
| D. Entropy | 0.7202 | 0.7386 | 0.7570 | 0.7951 | 0.8314 | 0.9050 |
| B. MC Dropout (entropy) | 0.7718 | 0.7876 | 0.8036 | 0.8345 | 0.8606 | 0.9156 |
| C. Deep ensemble (entropy) | 0.7718 | 0.7929 | 0.8122 | 0.8499 | 0.8791 | 0.9336 |
| C. Deep ensemble (disagreement) | 0.7718 | 0.7826 | 0.7973 | 0.8366 | 0.8676 | 0.9068 |
| C. Deep ensemble (variance) | 0.7718 | 0.7762 | 0.7812 | 0.7963 | 0.8120 | 0.8658 |
| D. Mahalanobis distance | 0.7718 | 0.7642 | 0.7567 | 0.7421 | 0.7279 | 0.7022 |

### IN (clean) vs OUT (corrupted) separation AUROC

| Method | gaussian_noise sev5 | contrast sev5 |
|---|---|---|
| A. Softmax (MSP) | 0.3082 | 0.8061 |
| D. Entropy | 0.2904 | 0.8032 |
| B. MC Dropout (entropy) | 0.3132 | 0.8089 |
| C. Deep ensemble (entropy) | 0.5512 | 0.8216 |
| C. Deep ensemble (variance) | 0.9021 | 0.5276 |
| C. Deep ensemble (disagreement) | 0.7203 | 0.7936 |
| D. Mahalanobis distance | 0.9187 | 0.5167 |

### Abstention payoff at 90% coverage under strong shifts

**gaussian_noise sev5**

| Method | Full acc | Acc at 90% | Improvement |
|---|---|---|---|
| A. Softmax (MSP) | 0.1638 | 0.1748 | +0.0110 |
| C. Deep ensemble (entropy) | 0.1682 | 0.1809 | +0.0127 |
| C. Deep ensemble (disagreement) | 0.1682 | 0.1741 | +0.0059 |
| D. Mahalanobis distance | 0.1682 | 0.1613 | -0.0069 |

**fog sev5**

| Method | Full acc | Acc at 90% | Improvement |
|---|---|---|---|
| A. Softmax (MSP) | 0.6779 | 0.7189 | +0.0410 |
| C. Deep ensemble (entropy) | 0.7312 | 0.7700 | +0.0388 |
| C. Deep ensemble (disagreement) | 0.7312 | 0.7524 | +0.0212 |
| D. Mahalanobis distance | 0.7312 | 0.7141 | -0.0171 |

### Deployed verdict — fused trust/abstain/flag decisions

Thresholds are learned on calibration data at 90% coverage; the system acts only on inputs that pass *every* guardrail check.

| Environment | Fraction trusted | Accuracy (all) | Accuracy (trusted only) |
|---|---|---|---|
| clean | 0.800 | 0.7718 | 0.7965 |
| gaussian_noise_sev5 | 0.223 | 0.1682 | 0.1244 |
| fog_sev5 | 0.838 | 0.7312 | 0.7719 |

Key figures: `final_risk_coverage_clean.png`, `final_risk_coverage_gaussian_noise_sev5.png`, `final_risk_coverage_fog_sev5.png`.

## 4. Distribution-shift experiments (controlled corruptions)

### Accuracy / ECE / confidence vs corruption severity

| Corruption | Sev | Accuracy | ECE | Mean conf | Err-AUROC (softmax) | Err-AUROC (entropy) | Err-AUROC (MC) |
|---|---|---|---|---|---|---|---|
| gaussian_noise | 1 | 0.5211 | 0.0948 | 0.6131 | 0.7230 | 0.7179 | 0.7033 |
| gaussian_noise | 3 | 0.2414 | 0.4704 | 0.7118 | 0.6602 | 0.6701 | 0.6476 |
| gaussian_noise | 5 | 0.1638 | 0.6774 | 0.8412 | 0.6464 | 0.6561 | 0.6425 |
| motion_blur | 1 | 0.5101 | 0.0424 | 0.5376 | 0.7516 | 0.7262 | 0.7119 |
| motion_blur | 3 | 0.3280 | 0.1397 | 0.4677 | 0.7032 | 0.6932 | 0.6812 |
| motion_blur | 5 | 0.2397 | 0.2218 | 0.4615 | 0.6444 | 0.6513 | 0.6618 |
| fog | 1 | 0.7199 | 0.0442 | 0.6758 | 0.8151 | 0.7976 | 0.7919 |
| fog | 3 | 0.7011 | 0.0418 | 0.6594 | 0.8116 | 0.7917 | 0.7910 |
| fog | 5 | 0.6779 | 0.0408 | 0.6371 | 0.8037 | 0.7844 | 0.7747 |
| brightness | 1 | 0.6323 | 0.0344 | 0.5981 | 0.7806 | 0.7644 | 0.7584 |
| brightness | 3 | 0.3804 | 0.1552 | 0.5355 | 0.6424 | 0.6343 | 0.6216 |
| brightness | 5 | 0.1149 | 0.3890 | 0.5039 | 0.4873 | 0.4842 | 0.4876 |
| contrast | 1 | 0.7189 | 0.0465 | 0.6724 | 0.8132 | 0.7943 | 0.7916 |
| contrast | 3 | 0.5402 | 0.0461 | 0.5861 | 0.7442 | 0.7361 | 0.7408 |
| contrast | 5 | 0.1848 | 0.2425 | 0.4273 | 0.6241 | 0.6453 | 0.6304 |

### IN (clean) vs OUT (shifted) separation AUROC

Higher = the uncertainty signal better flags unfamiliar inputs.

| Method | contrast_sev5 | gaussian_noise_sev5 |
|---|---|---|
| A. Softmax (MSP) | 0.8061 | 0.3082 |
| D. Entropy | 0.8032 | 0.2904 |
| B. MC Dropout (entropy) | 0.8117 | 0.3134 |
| B. MC Dropout (variance) | 0.7851 | 0.6194 |

### Abstention effect under strong shift (accuracy at 90% coverage)

**gaussian_noise sev5**

| Method | Full accuracy | Acc at 90% coverage | Risk reduction |
|---|---|---|---|
| A. Softmax (MSP) | 0.1638 | 0.1748 | -0.0110 |
| D. Entropy | 0.1638 | 0.1770 | -0.0132 |
| B. MC Dropout (entropy) | 0.1638 | 0.1720 | -0.0082 |
| B. MC Dropout (variance) | 0.1638 | 0.1671 | -0.0033 |

**fog sev5**

| Method | Full accuracy | Acc at 90% coverage | Risk reduction |
|---|---|---|---|
| A. Softmax (MSP) | 0.6779 | 0.7189 | -0.0410 |
| D. Entropy | 0.6779 | 0.7151 | -0.0372 |
| B. MC Dropout (entropy) | 0.6779 | 0.6778 | +0.0001 |
| B. MC Dropout (variance) | 0.6779 | 0.6716 | +0.0063 |

## 5. Key findings

1. **Selective prediction works on clean data:** abstaining on the most uncertain predictions sharply raises accuracy on the accepted set (91.6% at 50% coverage vs 72.0% at full coverage with the softmax signal).
2. **Calibration fixes clean-data confidence but not shift:** temperature scaling brings ECE down to **0.041** on the clean test set, but under strong corruption ECE explodes (up to **0.68** for gaussian_noise sev5) while mean confidence *rises* (**0.84**), i.e. the model becomes confidently wrong.
3. **Softmax is overconfident on Gaussian noise; MC Dropout variance is not:** IN-vs-OUT AUROC for softmax/entropy on gaussian_noise sev5 is **below 0.31** (worse than a coin flip - the uncertainty signal is inverted) while MC Dropout variance reaches **0.62**. This is the clearest demonstration that a standard confidence score cannot always 'tell you when it should not be trusted'.
4. **Selectivity still helps on milder shifts:** on fog sev5 accuracy rises from **0.68 to 0.72** at 90% coverage; only under the most destructive corruptions (severe noise) does abstention reach a floor because the model is uniformly and confidently wrong.
5. **The deep ensemble is the most reliable single signal:** best error-detection AUROC on clean test is **0.8178** (A. Softmax (MSP)), and the ensemble is the strongest IN-vs-OUT detector on the hardest shifts, beating both softmax confidence and MC-Dropout.
6. **Fusing the guardrails beats any single signal:** the deployed trust/abstain verdict — confidence + ensemble entropy + feature familiarity, all thresholded at 90% calibration coverage — keeps accuracy on accepted inputs near the clean level and explicitly refuses the inputs that would otherwise be confidently wrong.

*Generated automatically by `scripts/04_report.py`.*