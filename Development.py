"""ML-T2-016 — Development stage driver.

Runs the full development pipeline for "Teaching a Machine Learning System
When Not to Trust Its Own Prediction":

    python Development.py all            # train -> evaluate -> shift -> final -> report
    python Development.py train          # step 1: train the baseline CNN
    python Development.py evaluate       # step 2: clean-set calibration + eval
    python Development.py shift          # step 3: distribution-shift experiments
    python Development.py final          # step 4: ensemble + Mahalanobis + fusion eval
    python Development.py report         # step 5: results/report.md
    python Development.py demo           # step 6: launch the TrustGuard dashboard
"""
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

STEPS = {
    "train": "01_train_baseline.py",
    "evaluate": "02_evaluate_clean.py",
    "shift": "03_shift_experiments.py",
    "final": "05_final_evaluation.py",
    "report": "04_report.py",
}


def run(script, extra=None):
    extra = extra or []
    cmd = [sys.executable, os.path.join(BASE, "scripts", script), *extra]
    print("$ " + " ".join(cmd))
    rc = subprocess.run(cmd)
    if rc.returncode != 0:
        sys.exit(rc.returncode)


def demo():
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    os.path.join(BASE, "app", "app.py")])


def main(argv):
    if not argv:
        print(__doc__)
        return
    step = argv[0].lower()
    if step == "all":
        for name in ("train", "evaluate", "shift", "final", "report"):
            run(STEPS[name])
        print("\nPipeline complete. See results/report.md")
    elif step == "demo":
        demo()
    elif step in STEPS:
        run(STEPS[step], argv[1:])
    else:
        print(f"Unknown step '{step}'. Options: {list(STEPS)} | all | demo")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])