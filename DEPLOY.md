# Deploying TrustGuard to a public URL

The repository is already deploy-ready. Two routes give you the
**Demo Video Link / Deployment URL** the submission form asks for.

---

## Route A (recommended): Streamlit Community Cloud — free, public, permanent URL

This is a real cloud deployment, so the evaluation team can click a link and
interact with the dashboard themselves. It is the strongest option for the
"Deployment URL" field.

**One-time setup (about 3 minutes):**

1. Go to <https://share.streamlit.io> and click **Continue with GitHub**.
   Authorise Streamlit to read your public repositories.
2. Click **New app** → **Deploy a new app**.
3. Fill in exactly:

   | Field | Value |
   |---|---|
   | Repository | `madiredypalvasha-06/ML-T2-016-TrustGuard` |
   | Branch | `main` |
   | Main file path | `app/app.py` |
   | Python version | `3.11` |

4. Click **Deploy**. Watch the logs; the first build installs PyTorch
   (~800 MB) and takes 3–6 minutes. Two things to expect in the log, both
   normal:
   * `cifar-10-python.tar.gz` downloading — CIFAR-10 is fetched on first use
     by `torchvision` and is intentionally not committed to the repository.
   * `Creating wheel for torch ... done` — the slow step.
5. When it finishes you get a URL of the form
   `https://ml-t2-016-trustguard.streamlit.app`. **That is your demo URL.**

**Notes**

* The four model checkpoints, the fitted temperature and the Mahalanobis
  statistics are all committed under `checkpoints/`, so the deployed app uses
  exactly the weights the paper reports. No retraining is needed.
* If the app cold-starts slowly the first time (CIFAR-10 download), reload
  once. Subsequent loads are fast.
* Free tier gives 1 GB RAM, which is ample here: the whole ensemble is
  4 x 2.2 M parameters.
* If you would rather not publish a public app, skip to Route B.

---

## Route B: Demo video on Google Drive

If you would rather not publish the app publicly, upload the recording and
share the link.

1. Go to <https://drive.google.com> → **New** → **File upload** →
   `TrustGuard_demo_walkthrough.mp4` (in `submission/`).
2. Right-click the uploaded file → **Share** →
   *General access* → **Anyone with the link** → **Viewer** → **Copy link**.
3. Change `https://drive.google.com/file/d/FILE_ID/view` to
   `https://drive.google.com/uc?export=download&id=FILE_ID`,
   or simply use the "preview" link — both open in a browser.

**Verify before you paste the link.** Open it in a private/incognito window and
confirm the video actually plays. A Google Drive link that requires sign-in, or
where the owner has to change permissions first, will fail for the evaluation
team.

---

## What the video shows

`submission/TrustGuard_demo_walkthrough.mp4` (~3 min) walks through all four
instrument panels in the order that tells the project's story:

| # | Panel | What it demonstrates |
|---|---|---|
| 1 | Mission Control — clean input | A `TRUST` verdict with all three guardrails green |
| 2 | Mission Control — corrupted input | The verdict flips to `ABSTAIN`/`REVIEW`; the familiarity guardrail goes red |
| 3 | Drift Monitor — gaussian noise, severity 2 | The `WATCHPOINT` alarm: 70.2% of inputs still accepted, but only 38.4% accurate on them |
| 4 | Calibration Lab | Raw softmax over-confidence vs the temperature-scaled fit |
| 5 | Method Showdown | All seven uncertainty signals head-to-head |

Step 3 is the centrepiece — it is the "silent failure" case, where a naive
classifier's confidence looks healthy while its accuracy has collapsed.

---

## Troubleshooting

**`torch` wheel build fails.** Free-tier builds occasionally time out on the
PyTorch install. Redeploy, or pin a smaller wheel in `requirements.txt`
(`torch==2.4.1+cpu`) and add a `packages.txt` if a system library is missing.

**App shows "engine revving up" for minutes on first load.** That is the
CIFAR-10 download plus the first threshold fit. Reload after a minute.

**`FileNotFoundError` for a checkpoint.** The entrypoint path is wrong. It must
be `app/app.py`, not `app.py` — the app lives in a subdirectory.
