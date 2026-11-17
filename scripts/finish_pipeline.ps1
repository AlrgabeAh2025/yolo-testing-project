# Run after raw images/masks/metadata are ready.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\finish_pipeline.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Pip = Join-Path $Root ".venv\Scripts\pip.exe"
$WheelDir = Join-Path $Root "data\raw\_download\wheels"

Write-Host "== Install torch from local wheels if present =="
$torch = Get-ChildItem $WheelDir -Filter "torch-*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
$vision = Get-ChildItem $WheelDir -Filter "torchvision-*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($torch -and $vision -and -not (Test-Path (Join-Path $torch.FullName + ".aria2")) -and -not (Test-Path (Join-Path $vision.FullName + ".aria2"))) {
    & $Pip install --no-index --find-links $WheelDir $torch.FullName $vision.FullName
    & $Pip install ultralytics
} else {
    Write-Host "Torch wheels not ready yet; skipping CUDA install for now."
}

Write-Host "== Prepare YOLO dataset =="
& $Py scripts\prepare_yolo_dataset.py

Write-Host "== Verify dataset =="
& $Py scripts\verify_dataset.py

Write-Host "== CUDA check =="
& $Py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

Write-Host "Done."
