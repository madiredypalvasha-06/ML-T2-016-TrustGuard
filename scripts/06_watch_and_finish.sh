#!/bin/bash
# Watch for all ensemble member checkpoints, then finish the pipeline:
#   1) scripts/05_final_evaluation.py   (full A/B/C/D + fused-verdict results)
#   2) scripts/04_report.py             (regenerate results/report.md)
# Launch:  nohup bash scripts/06_watch_and_finish.sh > logs/watch_finish.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1

NEEDED=(checkpoints/member_0.pt checkpoints/member_1.pt \
        checkpoints/member_2.pt checkpoints/member_3.pt)

while :; do
  ready=1
  for f in "${NEEDED[@]}"; do
    [ -f "$f" ] || ready=0
  done
  if [ "$ready" = "1" ]; then
    echo "[watch] all members present -> running final evaluation"
    python3 -u scripts/05_final_evaluation.py --members 4 && \
      python3 -u scripts/04_report.py
    echo "[watch] done"
    exit 0
  fi
  echo "[watch] waiting for ensemble members ..."
  sleep 90
done