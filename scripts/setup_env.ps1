# Setup GPU training environment for YOLO skin-cancer project.
# Run from project root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvPip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtualenv..."
    python -m venv .venv
}

Write-Host "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing PyTorch with CUDA 12.4..."
# Prefer aria2 for the large CUDA wheel when available (faster resume / multi-connection)
$Aria = Join-Path $ProjectRoot "tools\aria2\aria2-1.37.0-win-64bit-build1\aria2c.exe"
$WheelDir = Join-Path $ProjectRoot "data\raw\_download\wheels"
New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null
if (Test-Path $Aria) {
    Write-Host "Downloading torch/torchvision wheels via aria2..."
    $TorchUrl = "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl"
    $VisionUrl = "https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp312-cp312-win_amd64.whl"
    & $Aria -c -x 16 -s 16 -k 1M --file-allocation=none -d $WheelDir -o "torch-2.6.0+cu124-cp312-cp312-win_amd64.whl" $TorchUrl
    & $Aria -c -x 16 -s 16 -k 1M --file-allocation=none -d $WheelDir -o "torchvision-0.21.0+cu124-cp312-cp312-win_amd64.whl" $VisionUrl
    & $VenvPip install --default-timeout=1000 (Join-Path $WheelDir "torch-2.6.0+cu124-cp312-cp312-win_amd64.whl") (Join-Path $WheelDir "torchvision-0.21.0+cu124-cp312-cp312-win_amd64.whl")
} else {
    & $VenvPip install --default-timeout=1000 torch torchvision --index-url https://download.pytorch.org/whl/cu124
}

Write-Host "Installing project requirements..."
& $VenvPip install -r requirements.txt

Write-Host "`nVerifying CUDA..."
& $VenvPython -c @"
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')
"@

Write-Host "`nDone. Activate with: .\.venv\Scripts\Activate.ps1"
