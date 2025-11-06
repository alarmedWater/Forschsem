<#
Setup YOLOv8 CUDA-Only Env mit Miniforge/Conda unter Windows.

Erstellt/erneuert:
  - Env: yolo8-seg
  - Python 3.10
  - PyTorch 2.1.2 + CUDA 12.1
  - torchvision, torchaudio
  - numpy < 2, pandas, opencv, pillow, requests, pyyaml
  - ultralytics + ONNX-Stack + Albumentations + thop

Aufruf (z.B. aus Miniforge Prompt / CMD):
  powershell -NoProfile -ExecutionPolicy Bypass -File .\pipeline_yolo8_cuda.ps1 -force

Parameter:
  -force  -> existierende 'yolo8-seg' Umgebung vorher löschen
  -help   -> Hilfe anzeigen
#>

[CmdletBinding()]
param(
    [switch]$force,
    [switch]$help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$EnvName       = "yolo8-seg"
$PythonVersion = "3.10"

if ($help) {
    Write-Host "Usage: .\pipeline_yolo8_cuda.ps1 [-force]"
    Write-Host "Always creates CUDA-enabled YOLOv8 env in '$EnvName'."
    exit 0
}

Write-Host "[YOLOv8] CUDA-ONLY SETUP"
Write-Host "[YOLOv8] ENV=$EnvName, PYTHON=$PythonVersion, FORCE=$($force.IsPresent)"
Write-Host ""

# --- Conda prüfen ---
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "[YOLOv8] ❌ 'conda' nicht gefunden. Bitte Miniforge/Anaconda Prompt (CMD oder PowerShell) nutzen oder 'conda init' ausführen."
    exit 1
}

function New-Or-ReuseEnv {
    param(
        [string]$Name,
        [string]$PyVer,
        [bool]$Force
    )

    $exists = conda env list | Select-String "^\s*$Name\s"
    if ($exists) {
        if ($Force) {
            Write-Host "[YOLOv8] Removing existing env '$Name' (force)..."
            conda env remove -n $Name -y | Out-Null
        }
        else {
            Write-Host "[YOLOv8] Env '$Name' exists. Reusing."
            return
        }
    }

    Write-Host "[YOLOv8] Creating env '$Name' (Python $PyVer)..."
    conda create -n $Name -c conda-forge python=$PyVer -y | Out-Null
}

function Install-PyTorchCudaStack {
    param([string]$Name)

    Write-Host "[YOLOv8] Installing PyTorch CUDA stack into '$Name'..."
    conda install -n $Name -y -c pytorch -c nvidia `
        pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=12.1 | Out-Null
}

function Install-BaseStack {
    param([string]$Name)

    Write-Host "[YOLOv8] Installing base scientific stack into '$Name'..."
    conda install -n $Name -y -c conda-forge `
        "numpy<2" pandas pyyaml opencv pillow requests | Out-Null
}

# --- Pipeline ---
New-Or-ReuseEnv        -Name $EnvName -PyVer $PythonVersion -Force:$force
Install-PyTorchCudaStack -Name $EnvName
Install-BaseStack        -Name $EnvName

Write-Host "[YOLOv8] Installing pip dependencies (Ultralytics & ONNX stack)..."
conda run -n $EnvName python -m pip install --no-cache-dir -U `
    "ultralytics==8.3.223" `
    "pycocotools>=2.0.0,<3.0.0" `
    "onnx==1.19.1" `
    "onnxruntime==1.16.3" `
    "onnxslim==0.1.72" `
    "albumentations==1.4.24" `
    "thop==0.1.1.post2209072238" | Out-Null

Write-Host ""
Write-Host "[YOLOv8] ✅ CUDA Environment '$EnvName' bereit."
Write-Host "Nutzen mit:"
Write-Host "  conda activate $EnvName"
Write-Host "  python C:\Users\fgerz1\Documents\JuliaFS\Forschsem\SegemtationTrainig_v1\models\train_yolo8s.py"
Write-Host ""
Write-Host "⚠️ Wichtig:"
Write-Host "  - In dieses Env KEIN TensorFlow installieren (sonst numpy/ABI-Chaos)."
Write-Host "  - Stelle sicher, dass NVIDIA-Treiber + CUDA Runtime zu pytorch-cuda=12.1 passen."
