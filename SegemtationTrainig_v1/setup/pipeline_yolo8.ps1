param(
    [switch]$cpu,
    [switch]$cuda,
    [switch]$force,
    [switch]$help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Defaults
$PY310    = "3.10"
$MODE     = "cpu"       # cpu | cuda
$ENV_NAME = "yolo8-seg"

if ($help) {
    Write-Host "Usage: .\pipeline_yolo8_setup.ps1 [-cpu|-cuda] [-force]"
    exit 0
}

# Argumente auswerten
if ($cpu)  { $MODE = "cpu" }
if ($cuda) { $MODE = "cuda" }

$forceFlag = $force.IsPresent
Write-Host "[YOLOv8] MODE=$MODE, FORCE=$forceFlag"

function Get-CondaEnvNames {
    conda env list |
        Where-Object { $_ -and ($_ -notmatch "^#") } |
        ForEach-Object {
            ($_ -split "\s+")[0]
        }
}

function Create-OrReuse {
    param(
        [string]$Name,
        [string]$PyVer,
        [bool]$Force
    )

    $envExists = Get-CondaEnvNames | Where-Object { $_ -eq $Name }

    if ($envExists) {
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
    conda create -y -n $Name -c conda-forge "python=$PyVer" | Out-Null
}

function Install-PytorchStack {
    param(
        [string]$Name,
        [string]$Mode
    )

    if ($Mode -eq "cuda") {
        Write-Host "[YOLOv8] Installing CUDA PyTorch stack into '$Name'..."
        conda install -y -n $Name -c pytorch -c nvidia `
            "pytorch=2.1.2" "torchvision=0.16.2" "torchaudio=2.1.2" "pytorch-cuda=12.1"
    }
    else {
        Write-Host "[YOLOv8] Installing CPU PyTorch stack into '$Name'..."
        conda install -y -n $Name -c pytorch `
            "pytorch=2.1.2" "torchvision=0.16.2" "torchaudio=2.1.2" "cpuonly"
    }
}

function Base-Science-Stack {
    param([string]$Name)

    Write-Host "[YOLOv8] Installing base science stack into '$Name'..."
    conda install -y -n $Name -c conda-forge `
        "numpy<2" "pandas" "pyyaml" "opencv" "pillow" "requests"
}

# Pipeline
Create-OrReuse -Name $ENV_NAME -PyVer $PY310 -Force:$forceFlag
Install-PytorchStack -Name $ENV_NAME -Mode $MODE
Base-Science-Stack -Name $ENV_NAME

Write-Host "[YOLOv8] Installing pip deps (Ultralytics & ONNX)..."
conda run -n $ENV_NAME pip install --no-cache-dir -U `
    "ultralytics==8.3.223" `
    "pycocotools>=2.0.0,<3.0.0" `
    "onnx==1.19.1" `
    "onnxruntime==1.16.3" `
    "onnxslim==0.1.72" `
    "albumentations==1.4.24" `
    "thop==0.1.1.post2209072238"

Write-Host ""
Write-Host "[YOLOv8] ✅ Environment '$ENV_NAME' is ready."
Write-Host "Use it with:"
Write-Host "  conda activate $ENV_NAME"
Write-Host "  python C:\Users\<deinUser>\Documents\JuliaFS\Forschsem\SegemtationTrainig_v1\models\train_yolo8s.py"
