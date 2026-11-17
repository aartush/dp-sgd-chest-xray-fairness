#!/usr/bin/env bash
# Resilient grid driver.
#
# `python -m src.runner --grid main` spawns a child per run but is itself one
# long-lived process; when that parent is killed the whole batch dies with it.
# This script inverts the nesting: the shell is the driver, every run is an
# independent short-lived interpreter, and a failure only affects that one run.
# Rerunning the script skips everything already on disk, so it is resumable.
#
# Usage:
#   bash scripts/run_grid.sh main
#   bash scripts/run_grid.sh imbalance
#   bash scripts/run_grid.sh chest
#   bash scripts/run_grid.sh all
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG_DIR=results
mkdir -p "$LOG_DIR"

run_cfg() { # dataset method eps seed [extra json]
  local dataset=$1 method=$2 eps=$3 seed=$4 extra=${5:-}
  local epsjson
  if [ "$eps" = "none" ]; then epsjson=null; else epsjson=$eps; fi
  local cfg
  cfg=$(printf '{"dataset":"%s","method":"%s","eps":%s,"seed":%s,"epochs":12,"batch_size":32,"lr":0.001,"weight_decay":0.0001%s}' \
        "$dataset" "$method" "$epsjson" "$seed" "$extra")
  local rid
  rid=$($PY -c "
import json,sys
sys.path.insert(0,'.')
from src.runner import run_id
from src.train import TrainConfig
print(run_id(TrainConfig(**json.loads(sys.argv[1]))))
" "$cfg" 2>/dev/null)
  if [ -f "$LOG_DIR/runs/$rid.json" ]; then
    echo "[skip] $rid $dataset $method eps=$eps seed=$seed"
    return 0
  fi
  echo "[run ] $rid $dataset $method eps=$eps seed=$seed"
  local t0=$SECONDS
  if $PY -m src.runner --single "$cfg" >> "$LOG_DIR/grid.log" 2>&1; then
    echo "       done in $((SECONDS-t0))s"
  else
    echo "       FAILED after $((SECONDS-t0))s (see results/grid.log)"
  fi
}

grid_main() {
  for method in baseline classweight pretrained pretrained+classweight; do
    for eps in none 1.0 3.0 8.0; do
      for seed in 0 1 2; do
        run_cfg pneumoniamnist "$method" "$eps" "$seed"
      done
    done
  done
}

grid_imbalance() {
  # ratio null = the native 1:2.88 split; 1.0 = balanced; 8.0 = severe
  for ratio in null 1.0 8.0; do
    local extra=",\"imbalance_ratio\":$ratio"
    for method in baseline classweight; do
      for eps in none 1.0 3.0; do
        for seed in 0 1; do
          run_cfg pneumoniamnist "$method" "$eps" "$seed" "$extra"
        done
      done
    done
  done
}

grid_chest() {
  for method in baseline classweight; do
    for eps in none 3.0; do
      run_cfg chestmnist "$method" "$eps" 0 ',"multilabel":true,"max_train":8000,"epochs":6'
    done
  done
}

case "${1:-main}" in
  main)      grid_main ;;
  imbalance) grid_imbalance ;;
  chest)     grid_chest ;;
  all)       grid_main; grid_imbalance; grid_chest ;;
  *) echo "unknown grid: $1"; exit 1 ;;
esac

echo "== summary =="
$PY -m src.runner --list | tail -5
