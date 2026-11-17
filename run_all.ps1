# Full reproduction on Windows PowerShell.
# From the project root:  powershell -ExecutionPolicy Bypass -File .\run_all.ps1
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "No venv found. Create it first:" -ForegroundColor Yellow
    Write-Host "  py -3.12 -m venv .venv"
    Write-Host "  .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
    Write-Host "  .venv\Scripts\python.exe -m pip install opacus==1.5.4 medmnist scikit-learn pandas matplotlib tqdm"
    exit 1
}

Write-Host "== environment ==" -ForegroundColor Cyan
& $py -c "import torch,opacus,medmnist; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); print('opacus',opacus.__version__,'medmnist',medmnist.__version__)"

Write-Host "== main grid ==" -ForegroundColor Cyan
& $py -u -m src.runner --grid main

Write-Host "== imbalance sweep ==" -ForegroundColor Cyan
& $py -u -m src.runner --grid imbalance

Write-Host "== chestmnist ==" -ForegroundColor Cyan
& $py -u -m src.runner --grid chest

Write-Host "== analysis ==" -ForegroundColor Cyan
& $py -m src.analyze

Write-Host "Done. See results\RESULTS.md" -ForegroundColor Green
