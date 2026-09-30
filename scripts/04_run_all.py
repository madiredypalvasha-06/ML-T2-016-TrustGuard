"""04 — Run the full pipeline end-to-end.

Usage: python scripts/04_run_all.py
"""
import os
import subprocess
import sys

SCRIPTS = ["01_train_baseline.py", "02_evaluate_clean.py",
           "03_shift_experiments.py", "04_report.py"]


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for s in SCRIPTS:
        print(f"\n{'='*70}\n>>> running {s}\n{'='*70}")
        rc = subprocess.run([sys.executable, os.path.join(base, "scripts", s)])
        if rc.returncode != 0:
            print(f"[run_all] FAILED at {s} (rc={rc.returncode})")
            sys.exit(rc.returncode)
    print("\n[run_all] pipeline complete. See results/")


if __name__ == "__main__":
    main()