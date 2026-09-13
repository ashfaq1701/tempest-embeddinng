#!/bin/bash
# Waits for ML-20M to be recorded, then hands off to WikiLink -> Yelp.
#
# The running driver holds a pre-edit inode listing ML-20M, Yelp, WikiLink, so
# editing the file cannot reorder what it does next -- it would start Yelp. This
# stops it once ML-20M is recorded and launches the reordered continuation.
set -u
WD=/its/home/ms2420/tempest-embeddinng
cd "$WD" || exit 1
D=$WD/logs/geometries_lorentz/run_3_seed3/DRIVER.log
W=$WD/logs/geometries_lorentz/run_3_seed3/REORDER.log
echo "[$(date '+%F %T')] waiting for ML-20M, then WikiLink -> Yelp" > "$W"
while true; do
  if grep -qE '^\[.*\] (DONE|FAIL|COLLAPSE) +ML-20M' "$D" 2>/dev/null; then
    echo "[$(date '+%F %T')] $(grep -E '^\[.*\] (DONE|FAIL|COLLAPSE) +ML-20M' "$D" | tail -1)" >> "$W"
    break
  fi
  pgrep -f "run_lorentz_sweep_seed3\.sh" > /dev/null || {
    echo "[$(date '+%F %T')] driver gone with no ML-20M record" >> "$W"; break; }
  sleep 60
done
for p in $(pgrep -f "run_lorentz_sweep_seed3\.sh"); do kill "$p" 2>/dev/null; done
sleep 3
for p in $(pgrep -f "train_link_property_prediction"); do kill "$p" 2>/dev/null; done
sleep 10
for p in $(pgrep -f "train_link_property_prediction"); do kill -9 "$p" 2>/dev/null; done
sleep 5
echo "[$(date '+%F %T')] primary driver stopped; launching WikiLink -> Yelp" >> "$W"
setsid nohup "$WD/scripts/run_lorentz_wikilink_yelp.sh" 3 3 > /dev/null 2>&1 < /dev/null &
sleep 20
echo "[$(date '+%F %T')] continuation pid: $(pgrep -f run_lorentz_wikilink_yelp | tr '\n' ' ')" >> "$W"
