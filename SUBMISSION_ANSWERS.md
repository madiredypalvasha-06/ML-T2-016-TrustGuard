# LearnDepth Track 2 — Final Submission Form: Ready-to-Paste Answers

**Project:** ML-T2-016 — *Teaching a Machine Learning System When Not to Trust Its Own Prediction*
**Repo:** <https://github.com/madiredypalvasha-06/ML-T2-016-TrustGuard>
**Prepared:** 30 September 2026

Everything below is either a value you supply or text you can paste verbatim.
Items marked ⚠️ **need your input** — I cannot know them.

---

## Uploads

| Form field | What to upload |
|---|---|
| **Technical Paper** (1 file, ≤10 MB) | `Technical_Paper_ML-T2-016.pdf` — 12 pages, 0.82 MB |
| **Final Project Screenshots** (up to 5, ≤10 MB each) | All 5 files in `submission/screenshots/` |

### Screenshot manifest

| # | File | Shows |
|---|---|---|
| 1 | `01_mission_control_trust.png` | Clean input → `TRUST`, all three guardrails green |
| 2 | `02_mission_control_abstain_ood.png` | Corrupted input → `ABSTAIN`/`REVIEW`, familiarity guardrail red |
| 3 | `03_drift_monitor_watchpoint.png` | `WATCHPOINT` alarm: 70.2% accepted but only 38.4% accurate |
| 4 | `04_calibration_lab.png` | Reliability: raw softmax vs temperature-scaled |
| 5 | `05_method_showdown.png` | All seven uncertainty signals head-to-head |

---

## Form fields

### Full Name
```
Palvasha Madireddy
```

### Email
```
madiredypalvasha@gmail.com
```
> Use the address you will still have access to after the internship.

### College & Department
```
Woxsen University — B.Tech (AIML), 3rd Year
```
> Woxsen University · B.Tech in Artificial Intelligence & Machine Learning (AIML) · Third year

### Project ID + Project Title
```
ML-T2-016 — Teaching a Machine Learning System When Not to Trust Its Own Prediction
```

### Project GitHub Repository
```
https://github.com/madiredypalvasha-06/ML-T2-016-TrustGuard
```

### Technical Paper
Upload `Technical_Paper_ML-T2-016.pdf`

### Demo Video Link / Deployment URL
✅ **Paste this (Option A — the live app):**

```
https://ml-t2-016-trustguard-e5kkubzkekljbgrw9fvzkb.streamlit.app
```

A live URL lets the evaluator interact with the system themselves, which is
much stronger than a video.

⚠️ Note the random suffix. Streamlit's free tier appends one to the app name,
so the URL is *not* the bare `ml-t2-016-trustguard.streamlit.app`; copy the
address from the browser or the dashboard rather than retyping it.

Before submitting, confirm two things:
* **Viewer access is Public** (dashboard → Settings). The app is private by
  default and a judge would hit a login wall.
* It loads in an **incognito window**. The owner is always signed in, which
  hides the login wall from you.

**Option B — if you'd rather submit the video.** Upload
`submission/TrustGuard_demo_walkthrough.mp4` (3:17, 5.3 MB) to Google Drive,
set link sharing to "Anyone with the link → Viewer", and paste that link.
Open it in an incognito window first to confirm it plays for someone not
signed in.

You can also submit **both** if the form allows only one: the URL in this
field, and the video in the video field if there is a separate one.

### Upload Final Project Screenshots
Upload all 5 PNGs listed in the manifest above.

---

### Briefly explain what your project solves and how it works.

```
A standard image classifier is judged on accuracy over data that looks like its
training set, which tells you nothing about how it behaves when the world
changes. In deployment the input stream drifts, and the network does not
degrade gracefully — it becomes confidently wrong. That is a silent-failure
hazard: a medical-triage or fraud-flag model reporting 95% confidence when its
true accuracy on that slice is 15% fails in a way no operator can see.

This project builds a trust-aware image classifier around a compact CIFAR-10 CNN
(2,196,810 parameters) that emits three things instead of one: a class
prediction, a calibrated reliability signal, and an explicit TRUST / REVIEW /
ABSTAIN verdict that a human can inspect and override.

The central claim is that "how sure is the model of its answer" and "how
familiar is this input" are different quantities, and you need both. I
implemented and benchmarked four families of uncertainty estimation on identical
data — maximum softmax probability with temperature scaling, MC Dropout (30
stochastic passes), a 4-member deep ensemble, and Mahalanobis feature-space
familiarity — producing seven comparable uncertainty scores, and evaluated each
along four reliability axes: calibration (ECE, reliability diagrams),
risk–coverage, error-detection AUROC, and in-distribution vs
out-of-distribution separation.

The evidence for fusion is direct. On the clean held-out test set the model
reaches 72.02% accuracy at ECE 0.0406 — well calibrated. Under severe Gaussian
noise the same checkpoint's mean confidence RISES to 0.84 while accuracy
collapses to 16.4% and ECE explodes to 0.68. Critically, the softmax signal's
IN-vs-OUT AUROC is 0.3082 — worse than a coin flip, i.e. actively inverted.
The very same inputs are caught almost perfectly by Mahalanobis distance
(0.9187) and deep-ensemble variance (0.9021). No single score wins everywhere:
Mahalanobis is the clearest illustration — it is the worst error detector on
clean data (0.3631) and the best under shift.

So the final system fuses calibrated confidence, ensemble entropy and feature
familiarity into a single three-way verdict with human-readable reasons
(`confidence=LOW · uncertainty=HIGH · familiarity=OOD(d=14.2)`). Thresholds are
fit on a held-out calibration split at a chosen coverage target rather than
hand-set. Deployed, it trusts 80% of clean inputs and is MORE accurate on what
it accepts (79.65%) than on the population (77.18%); under severe noise it
refuses ~78% of the stream; and in the most dangerous regime — moderate noise,
where a naive model still looks healthy — it raises a WATCHPOINT alarm showing
70.2% of inputs accepted but only 38.4% accurate on them. The deliverable is
not a better model but a control mechanism around it, exposed through a
Streamlit dashboard (TrustGuard) with four instrument panels so every claim can
be interrogated interactively.
```

---

### What was your contribution? Mention the modules/features you developed.

```
I developed the entire project end-to-end, from the research brief through the
model, the experiments, the report and the interactive system. Concretely:

RESEARCH DESIGN
- Wrote the Stage-1 research brief: reframed the assigned problem, selected
  CIFAR-10 with local corruption-based shift as the controlled testbed, and
  justified a four-family uncertainty taxonomy (A softmax+temperature, B
  MC-Dropout, C deep ensemble, D Mahalanobis) with an explicit evaluation plan.

DATA & EXPERIMENTAL INFRASTRUCTURE
- src/data.py — disciplined 45,000 train / 5,000 calibration / 10,000
  held-out-test split with a fixed permutation seed, so the test set is touched
  exactly once, at the end.
- src/corruptions.py — a from-scratch, deterministic, re-implementation of the
  five standard CIFAR-10-C corruption recipes (Gaussian noise, motion blur, fog,
  brightness, contrast) with the published per-severity parameter tables. I did
  this because the original CIFAR-10-C release is no longer downloadable, and
  it makes the shift experiments fully self-contained and reproducible.
- scripts/01–06 — a six-stage pipeline (train, clean evaluation, shift
  experiments, report generation, final evaluation, checkpoint watcher) with
  argparse configuration, per-epoch history logging, and a
  best-calibration-accuracy checkpoint rule.

MODELS & UNCERTAINTY
- src/models.py — a BatchNorm-free 7-layer CNN (2,196,810 parameters) with a
  feature-extraction head. The absence of BatchNorm is deliberate: MC-Dropout
  requires `model.train()` at inference to enable *only* dropout, and BatchNorm
  would silently invalidate the estimator.
- src/uncertainty.py — temperature scaling fitted by LBFGS on the calibration
  split, MC-Dropout sampling with predictive entropy / variance / mutual
  information, and predictive-entropy scoring.
- src/ensembles.py — the deep ensemble (mean of temperature-scaled softmaxes,
  plus entropy, variance, mutual information and fraction-disagreement) and
  TrustAwareClassifier, the fused verdict engine.
- src/mahalanobis.py — per-class feature means with a Ledoit-Wolf-shrunk pooled
  covariance, scored as the minimum class Mahalanobis distance.
- src/metrics.py — every metric hand-implemented (15-bin ECE, Brier,
  risk–coverage curve, accuracy-at-coverage, rank-based AUROC, error-detection
  AUROC) plus reliability-diagram and risk–coverage plotting, so the definitions
  are explicit and auditable rather than library defaults.
- scripts/05_final_evaluation.py — the main experiment: 7 methods x clean set,
  5 corruptions x 3 severities, IN-vs-OUT AUROC, abstention payoff, and the
  deployed verdict evaluation.

ANALYSIS & REPORTING
- The full experimental analysis, and a 10-section technical report including
  the negative results and an explicit limitations section.

INTERACTIVE SYSTEM
- app/app.py — the TrustGuard dashboard: Mission Control (single-input trust
  dossier with three gauges and per-check verdicts), Drift Monitor (batch stream
  scanning with histogram inspection, a flagged-input gallery and a WATCHPOINT
  alarm), Calibration Lab (side-by-side reliability diagrams) and Method
  Showdown (head-to-head comparison of all seven signals).
- app/engine.py — the runtime layer: cached resource loading, automatic profile
  selection, threshold fitting at a chosen coverage target, and batch scanners.
- app/theme.py — a custom dark instrument-panel design system.
- scripts/render_paper_pdf.py — a headless-Chromium renderer that typesets the
  technical paper to a paginated, figure-embedded PDF.
- scripts/capture_demo.py — a Playwright harness that drives the live dashboard
  to capture the submission screenshots and the demo walkthrough.

WHAT WAS GIVEN VS BUILT BY ME: the problem statement and the research problem
set were provided by LearnDepth Academy. Everything above — every module,
every experiment, every figure, the report and the dashboard — is my own work.
```

---

### What was the most difficult technical problem you faced, and how did you solve it?

```
The hardest problem was that the two quantities you would assume are
interchangeable — confidence and familiarity — are not, and my first instinct
was to build a single-score abstention system. I had to abandon that.

The specific difficulty: temperature scaling fixed calibration beautifully
(ECE 0.0406 on clean data) and I expected that to carry over. It did not. Under
severe Gaussian noise the model's mean confidence ROSE to 0.84 while accuracy
fell to 16.4%. I initially assumed I had a bug — perhaps the corruption was
misapplied, or the temperature was being applied in the wrong place. I spent
time ruling those out before accepting that the behaviour was real.

The decisive step was building the IN-vs-OUT AUROC measurement: score a matched
set of clean and corrupted inputs with each uncertainty signal and measure how
well it separates them. That produced the project's central result — softmax
AUROC 0.3082, i.e. worse than chance and actively pointing the wrong way — and
it explained the anomaly. A network that is uniformly wrong on a shifted input
produces a confident, wrong output, and softmax confidence says nothing about
whether the *input* resembles anything it was trained on. Post-hoc calibration
re-scales confidence; it cannot manufacture familiarity.

Having established that, the problem became a design problem: build a
complementary signal that measures input familiarity rather than answer
certainty. That is what the Mahalanobis and ensemble-variance scores do, and
they detect exactly the inputs that fool confidence (0.9187 and 0.9021). The
fused TrustAwareClassifier follows directly from that evidence.

A second, smaller difficulty was a subtle correctness trap in MC-Dropout: the
obvious CNN recipe includes BatchNorm, but BatchNorm's running statistics
update on every forward pass in train mode, so `model.train()` at inference
silently injects nondeterminism that is not dropout and invalidates the
stochastic estimator. I removed BatchNorm entirely and documented the accuracy
cost as an accepted trade-off rather than an oversight.

A third was a result I chose to report rather than hide: under severe noise,
accuracy on trusted inputs (0.1244) came out LOWER than on the stream as a
whole (0.1682). That looked like an embarrassing bug. It is not — when a
corruption destroys the signal in every image there is no good subset to select,
and only the aggregate trust rate carries information. I documented it as a
limitation and used it to reframe the deployment guidance around stream-level
escalation rather than per-input rescue.
```

---

### What technologies/tools did you actually use?

```
LANGUAGES & CORE LIBRARIES
- Python 3.11
- PyTorch 2.4 (torchvision for the CIFAR-10 loader) — model, training, inference
- NumPy — tensors, quantiles, linear algebra
- scikit-learn — LedoitWolf covariance estimator; used deliberately only here,
  with every evaluation metric implemented by hand
- matplotlib — reliability diagrams, risk–coverage curves, training curves
  (Agg backend, so it works headless)

UNCERTAINTY / RELIABILITY TECHNIQUES IMPLEMENTED FROM THE LITERATURE
- Maximum Softmax Probability (Hendrycks & Gimpel 2017)
- Temperature scaling, fitted with torch.optim.LBFGS (Guo et al. 2017)
- MC Dropout with mutual information (Gal & Ghahramani 2016)
- Deep Ensembles (Lakshminarayanan et al. 2017)
- Mahalanobis / least-confidence-outlier feature distance (Lee et al. 2018)
- Selective classification and risk–coverage analysis (El-Yaniv & Wiener 2010)
- Ledoit–Wolf covariance shrinkage (Ledoit & Wolf 2004)

APP & INTERACTION
- Streamlit 1.55 — the TrustGuard dashboard
- Plotly — confidence/entropy/familiarity gauges, distributions, AUROC bars
- A custom CSS design system (app/theme.py); Space Grotesk + IBM Plex Mono

TOOLING I BUILT OR USED FOR DELIVERY
- Playwright + headless Chromium — to typeset the technical paper to PDF
  (scripts/render_paper_pdf.py), to capture the submission screenshots, and to
  record the demo walkthrough (scripts/capture_demo.py)
- MathJax 3 — LaTeX math typesetting in the rendered paper
- Git + GitHub CLI — version control and repository publication
- ffmpeg — video encoding
- Apple MPS backend for inference (src/config.py auto-selects mps/cuda/cpu)

WHAT I DID NOT USE
- No paid or closed-source tooling, no cloud credits, no pretrained weights —
  the whole project is zero-cost and runs on a laptop, as the brief required.
  The four checkpoints are trained from scratch on CIFAR-10.
```

---

### What would you improve if you had another 30 days?

```
In priority order:

1. FIX THE ENSEMBLE AND GET ERROR BARS. Member 2 was warm-started from the
   baseline to fit my compute budget, so members 0 and 2 share a training
   trajectory and are correlated — which understates ensemble disagreement as an
   uncertainty signal, so the ensemble-variance numbers are conservative lower
   bounds. I would train all four from scratch and report means with confidence
   intervals over five seeds. Every number in the paper is currently single-run,
   and that is the weakest methodological point in the work.

2. ATTACK THE CONTRAST FAILURE. Mahalanobis and ensemble variance score 0.92 and
   0.90 on Gaussian noise but collapse to near-chance (0.517, 0.528) on severe
   contrast — a corruption that desaturates the image but leaves the object
   intact, so features stay near their class clusters. This is the most
   scientifically interesting loose end: my familiarity guardrail has a
   documented blind spot. I would test per-class covariances, a directional
   Mahalanobis, and explicitly training the detector on the corruption family it
   is meant to catch.

3. REPLACE THE COVERAGE QUANTILE WITH A RISK BOUND. The fused threshold uses
   per-signal quantiles at a target coverage, which is an approximation when
   combining three jointly non-monotonic scores. A Geifman–El-Yaniv
   risk-controlled bound would let the system GUARANTEE a risk level rather than
   merely target an acceptance rate — the property an actual deployment needs.

4. BROADEN THE OOD SUITE. Two shift families is not enough evidence. I would add
   SVHN and CIFAR-100 as genuine natural-OOD sets, plus the remaining CIFAR-10-C
   corruptions, and — most importantly — report a generalisation matrix showing
   which detector survives which shift, rather than the two-shift comparison I
   have.

5. CLOSE THE ENGINEERING GAP. No automated tests, no CI, no data-versioning. I
   would add unit tests for the metrics against reference implementations (this
   matters most, since I hand-rolled every metric), a Makefile, and a
   reproducible environment lockfile.

6. ARCHITECTURAL HONESTY. Add a BatchNorm variant with a separate
   MC-regularised head so the accuracy comparison is fair, and test a modern
   backbone to check that the qualitative conclusions are not an artefact of a
   small 2024-era CNN.
```

---

### What are the top 3 technical skills you improved during this internship?

```
1. UNCERTAINTY ESTIMATION AND CALIBRATION — the core skill.
   I moved from treating softmax probability as a confidence number to
   understanding it as one weak, distribution-dependent proxy among several.
   Concretely I implemented and then had to reason about the failure boundaries
   of temperature scaling, MC Dropout, deep ensembles and Mahalanobis distance,
   and learned that the interesting content is in the regime where each one
   breaks. Understanding WHY calibration fixes clean data but is powerless under
   shift — because it re-scales a number, and no rescaling can create familiarity
   with an input — is the single most useful thing I learned in the internship.

2. EXPERIMENTAL DESIGN AND EVALUATION DISCIPLINE.
   Building the splits so the test set is touched once, fitting temperature,
   Mahalanobis statistics and thresholds only on held-out calibration data,
   defining success criteria before running anything, and — most importantly —
   measuring the thing that actually matters. I had to design the IN-vs-OUT
   AUROC experiment to get past the anomaly in §1; without that measurement I
   would have "fixed" a real finding as if it were a bug. I also learned to
   hand-write every metric so its definition is explicit, and to report negative
   results (the 0.1244 trusted-subset accuracy, Mahalanobis being the worst
   clean-data error detector) instead of quietly dropping them.

3. DEBUGGING SCIENTIFIC SOFTWARE — reading an implementation, not just running it.
   The MC-Dropout/BatchNorm interaction, the inverted-AUROC anomaly, the
   warm-started ensemble member, the sign-inverted field name in a results JSON,
   and a Streamlit DOM race condition in my own capture script were all found by
   constructing a measurement that would have exposed the bug, not by staring at
   code. Related practical skill: building the tooling I needed rather than
   working around missing tooling — a Playwright-based paper renderer with
   MathJax and a page-numbered print layout, and a browser harness that drives
   the dashboard into specific states to capture reproducible evidence.
```

---

### What is one thing you can build or do now that you could not do before joining this internship?

```
Before the internship, if you handed me a model reporting 95% confidence, I would
have believed it. I knew accuracy and I knew cross-entropy; I did not know that
confidence could be a systematically misleading signal, and I would not have
known how to demonstrate it.

Now I can take any classifier and wrap it in a trustworthiness layer, and prove
whether the wrapper is needed. Concretely, I can now:
- Design and run the experiment that settles whether a model's confidence
  survives distribution shift — build matched in-distribution and shifted sets,
  score them with several uncertainty estimators, and measure separation with
  AUROC. I did this, and it is the reason this project has a finding instead of
  just an implementation: it revealed that the standard signal runs at 0.3082,
  worse than chance.
- Implement the four main uncertainty families and know their failure regimes
  well enough to predict which one will help in a given deployment.
- Turn a model into something with a calibrated reliability signal and a
  human-auditable abstention channel, then expose the whole thing as an
  interactive dashboard so a non-specialist can interrogate it.
- Reason about the actual operating decision — the coverage/escalation trade-off
  — as a business choice with a measurable cost, rather than as a model knob.

The generalisable skill: given any ML system, I can now identify where it fails
silently, measure that failure rather than assert it, and build the control
mechanism to catch it. That transfers to tabular models, NLP, and anything else
where a confident wrong answer is more expensive than a refusal.
```

---

### Which area do you want to learn next?
```
MLOps
```
> Select from: Advanced ML · Deep Learning · GenAI · MLOps · Data Science ·
> Full Stack · Cloud/DevOps · Research · Interview/Job Preparation · Other
>
> **Suggested, based on this project:** MLOps is the most natural next step —
> this project built the *detection* layer, and MLOps is where monitoring,
> drift detection and retraining pipelines get productionised. If you would
> rather stay closer to modelling, pick **Research** (uncertainty quantification
> is an active area and this project left clear open questions). Change the
> selection if your actual interest differs; it is a preference question, not a
> factual one.

---

### Overall, how would you rate your internship experience?
```
5
```

### What was the most valuable part of the internship?
```
The most valuable part was being given a problem where the obvious solution was
insufficient, and having the time and encouragement to find out why.

The assigned research question was "can an ML system recognise when it should not
trust its own prediction?" That is a question with a well-known answer
(uncertainty estimation, abstention, OOD detection), and the trap is to
reimplement the textbook methods, show that they help, and write it up. What
made this valuable instead was discovering that the textbook answer does not
generalise: the headline confidence signal is actively inverted on one of the
corruptions. I went in expecting to build a pipeline and came out having learned
how to design a measurement that can disprove the thing I was building — and
that reframing, from "implement the methods" to "find out when the methods lie,"
is the single most transferable thing I took away.

Three specific things made it work, and they are the reason I would rate this 5/5:

1. The problem was genuinely open. Nobody handed me a spec. I chose the dataset,
   defined the evaluation axes, decided the success criteria in the Stage-1 brief
   before writing any model code, and had to justify each choice. That is very
   different from a graded exercise, and it is much closer to research.

2. Zero-cost constraints were a feature, not a limitation. Being restricted to
   free tools and a laptop pushed me toward a small, efficient experimental
   design — a 2.2M-parameter CNN, deterministic locally-generated corruptions
   instead of a 2.6 GB download, hand-written metrics. Every constraint made the
   work more portable and more explainable.

3. The scope matched a real internship. Two to three weeks is enough to own a
   project end-to-end — data, model, experiments, analysis, report, and a
   working interactive system that someone else can actually use — without the
   illusion of depth. I finished with a deployed, inspectable artefact and a
   paper whose limitations section I believe rather than tolerate.

If I could change one thing it would be more time: the analysis kept expanding
past what three weeks allowed, and the two things I would do with more time
(seed-averaged error bars and the contrast-corruption failure) are exactly the
kind of rigor that turns a good project into a defensible result.
```

### What should LearnDepth improve for the next batch?
```
The programme is well structured — real research problems, a staged submission
format, and a genuine expectation of independent work. Four suggestions, offered
constructively:

1. TIGHTEN THE STAGE DEFINITIONS, AND SAY WHAT "DONE" LOOKS LIKE PER STAGE.
   The Stage-1 brief and the final submission form ask for overlapping
   material — a project description, a problem explanation, a contribution
   summary, evaluation plans — and it is not always clear which artefact is the
   authoritative one or how much detail each stage expects. A short rubric per
   stage (e.g. "Stage 1 is assessed on problem clarity and evaluation design
   only; implementation detail belongs to Stage 2") would remove a lot of
   guesswork about where to spend effort.

2. PROVIDE A STARTER REPOSITORY SKELETON.
   Spending early time on environment setup, project layout and a results-format
   convention is not where the learning is. A skeleton with the directory
   structure, a results-JSON schema, a metrics module and a requirements file
   with pinned versions would let every batch start on the actual research
   question on day one, and would also make submissions far easier to evaluate
   side by side.

3. GIVE MORE GUIDANCE ON REPRODUCIBILITY EXPECTATIONS.
   For a research-track internship, "your results must be independently
   verifiable" is a real requirement, but the expected bar is never stated. I
   ended up committing trained checkpoints (~42 MB) and all raw result JSON
   specifically so a reviewer could verify every number in my paper without
   retraining for 16 hours a member. Making that expectation explicit — and
   ideally normalising the result-file format — would raise the standard across
   the whole batch.

4. CLARIFY THE FINAL-FORM SUBMISSION WINDOW AND THE SOCIAL/REVIEW ITEMS.
   The exit form bundles substantive deliverables (paper, repository, demo,
   screenshots) with items that are not really deliverables — a Google review,
   a LinkedIn testimonial post. For a departing intern the review and the post
   are easy to overlook or to feel awkward about, and a deadline on the whole
   form creates avoidable last-minute risk. Separating the academic deliverables
   from the optional promotional items, and allowing the review to be submitted
   after the deadline, would help.

One further note, offered as a candidate improvement to the curriculum rather
than a complaint about it: the batch would benefit from a short session on
reading a paper's results sceptically — specifically, why a metric of 0.31 is
more interesting than one of 0.92. Interns reliably optimise for the best
number, and the most valuable habit in reliability work is noticing when a good
number means you tested the wrong thing. That instinct is what turned this
project's confusing anomaly into its central finding.
```

---

### Write a short testimonial about your LearnDepth internship experience.

```
I came into the LearnDepth Track 2 internship expecting to build a model and
report how accurate it was. I left having learned how to find out whether a
model's confidence can be trusted at all — which turned out to be a very
different question.

My project, "Teaching a Machine Learning System When Not to Trust Its Own
Prediction," started from an assigned research prompt with a familiar-looking
answer: implement uncertainty estimation and abstention. The valuable part was
discovering that the familiar answer does not survive contact with shifted data.
On clean inputs my model was well calibrated — 72% accuracy at an ECE of 0.04.
Under strong noise, the same checkpoint's confidence climbed to 0.84 while its
accuracy fell to 16%. When I built the measurement to understand why, the
standard confidence signal came out at 0.31 AUROC for detecting those inputs:
worse than a coin flip, actively pointing the wrong way. The inputs it missed
were caught almost perfectly by a feature-familiarity score instead. That result
is what turned the project from an implementation exercise into a finding, and
it is the reason the final system fuses several signals rather than trusting one.

What I valued most about the format was the freedom, and the responsibility that
came with it. Nobody gave me a specification. I chose the dataset, defined the
evaluation criteria in writing before writing any model code, and had to
defend every choice — including the results that did not flatter me. I reported
a case where the system performed worse on the inputs it trusted than on the
stream as a whole, and explained why that is the expected behaviour when a
corruption destroys the signal entirely, rather than quietly dropping it. The
programme also let me treat a laptop-only, zero-cost constraint as a design
problem, which pushed me toward a small efficient model, self-contained
corruption generation instead of a multi-gigabyte download, and every evaluation
metric written by hand so its definition was explicit.

I also got to finish with a real artefact rather than a report. I built
TrustGuard, a dashboard that turns the research into four instrument panels
where you can feed in a single image and see a TRUST / REVIEW / ABSTAIN verdict
with the reasoning, or sweep a corruption across a stream of images and watch the
system raise an alarm when it starts accepting inputs it gets wrong. Being able
to hand someone a link and say "try to break it" was the most satisfying part of
the whole project, and it made the limitations I had documented concrete instead
of abstract.

My honest advice to someone starting the next batch: do not treat the assigned
problem as a spec. The score you get for building the expected thing is much
lower than the score for noticing that the expected thing is wrong. The
momentum in this project came entirely from taking an anomaly seriously.

LearnDepth gave me a real problem, real autonomy, and exactly three weeks — a
combination that is genuinely hard to find, and which I would recommend to anyone
considering it.
```

---

### Kindly share the LinkedIn testimonial post link here
⚠️ **Your input.** Publish the testimonial above (trimmed to ~200–300 words for
LinkedIn) as a post, then paste the permalink here.

Post to LinkedIn → start a post → paste → **Publish**. Then open the post and
copy the URL from the address bar; it looks like
`https://www.linkedin.com/posts/<you>_trustguard-<id>`.

**Verify it first.** Open the link in an incognito window. LinkedIn frequently
serves a login wall to logged-out visitors, and a link that only works when
signed in will fail for the evaluation team. If it is blocked, either set the
post to "Public" visibility or paste the post text into this field instead and
write "link requires sign-in".

### Give a review on Google and upload the screenshot here
⚠️ **Your input.** Post a review at <https://share.google/z6zf6BIQIlGmnMUNS>,
take a screenshot showing the review and your name, and upload it here.

### Would you like to receive information about advanced LearnDepth programs?
```
Yes
```

### Would you be interested in future project/research opportunities with LearnDepth?
```
Yes
```

### GitHub Profile Link
```
https://github.com/madiredypalvasha-06
```

### LinkedIn Profile Link
⚠️ **Your input.** Your public profile URL, in the form
`https://www.linkedin.com/in/<your-name>/`.

I could not find this recorded in the project files and I will not guess at a
profile URL. Note that LinkedIn profile URLs are case-insensitive but the
custom part is yours to choose; copy it from the browser while logged in.

### Project Authenticity Declaration
```
Yes , I Confirm
```

### Reference & Attribution Declaration
```
Yes,  I confirm
```

### Final Submission Declaration
```
Yes , I Confirm
```

---

## Pre-submission checklist

- [x] College & Department: `Woxsen University — B.Tech (AIML), 3rd Year`
- [ ] Technical Paper PDF uploaded — `Technical_Paper_ML-T2-016.pdf`, 0.82 MB ✓
- [ ] All 5 screenshots uploaded from `submission/screenshots/` ✓
- [ ] Demo URL live **or** video link pasted and verified in an incognito window
- [ ] LinkedIn post published, link copied, and verified while logged out
- [ ] Google review posted, screenshot taken and uploaded
- [ ] LinkedIn profile URL pasted (⚠️)
- [ ] Every pasted answer proofread once for typos
- [ ] Click each link you pasted (repo, demo, LinkedIn, profiles) and confirm it
      opens — the form explicitly says links must be accessible
- [ ] The three declaration checkboxes ticked
