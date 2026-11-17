#!/usr/bin/env bash
# Full reproduction: setup check -> grids -> analysis.
# Run from the project root:  bash run_all.sh
set -euo pipefail
PY=.venv/Scripts/python.exe
if [ ! -f "$PY" ]; then
  echo "No venv at $PY. Create it first:"
  echo "  py -3.12 -m venv .venv"
  echo "  .venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
  echo "  .venv/Scripts/python.exe -m pip install opacus==1.5.4 medmnist scikit-learn pandas matplotlib tqdm"
  exit 1
fi

echo "== environment =="
$PY -c "import torch,opacus,medmnist; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); print('opacus',opacus.__version__,'medmnist',medmnist.__version__)"

echo "== main grid (48 runs) =="
$PY -u -m src.runner --grid main

echo "== imbalance sweep =="
$PY -u -m src.runner --grid imbalance

echo "== chestmnist scale check =="
$PY -u -m src.runner --grid chest

echo "== analysis =="
$PY -m src.analyze

echo "Done. See results/RESULTS.md and results/figures/"
