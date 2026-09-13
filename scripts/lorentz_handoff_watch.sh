#!/bin/bash
# Waits for Patent to finish in the primary sweep, then hands off to the 7-dataset
# continuation driver. The primary driver would otherwise start GoogleLocal without
# YouTube in its queue; it is killed once Patent is recorded.
set -u
WD=/its/home/ms2420/tempest-embeddinng
cd "$WD" || exit 1
DRIVER=$WD/logs/geometries_lorentz/run_3/DRIVER.log
WATCH=$WD/logs/geometries_lorentz/run_3/HANDOFF.log
echo "[$(date '+%F %T')] watching $DRIVER for Patent completion" > "$WATCH"
while true; do
  if grep -qE '^\[.*\] (DONE|FAIL)  Patent ' "$DRIVER" 2>/dev/null; then
    LINE=$(grep -E '^\[.*\] (DONE|FAIL)  Patent ' "$DRIVER" | tail -1)
    echo "[$(date '+%F %T')] Patent recorded: $LINE" >> "$WATCH"
    for p in $(pgrep -f "run_geometries_lorentz\.sh"); do kill "$p" 2>/dev/null; done
    sleep 3
    for p in $(pgrep -f "train_link_property_prediction"); do kill "$p" 2>/dev/null; done
    sleep 10
    for p in $(pgrep -f "train_link_property_prediction"); do kill -9 "$p" 2>/dev/null; done
    sleep 5
    echo "[$(date '+%F %T')] primary driver stopped; launching continuation (7 datasets, seed 42)" >> "$WATCH"
    setsid nohup "$WD/scripts/run_geometries_lorentz_rest.sh" 3 42 > /dev/null 2>&1 < /dev/null &
    sleep 20
    echo "[$(date '+%F %T')] continuation pid: $(pgrep -f run_geometries_lorentz_rest\.sh | tr '\n' ' ')" >> "$WATCH"
    exit 0
  fi
  if ! pgrep -f "run_geometries_lorentz\.sh" > /dev/null; then
    echo "[$(date '+%F %T')] primary driver gone with no Patent record; launching continuation anyway" >> "$WATCH"
    setsid nohup "$WD/scripts/run_geometries_lorentz_rest.sh" 3 42 > /dev/null 2>&1 < /dev/null &
    exit 0
  fi
  sleep 60
done
