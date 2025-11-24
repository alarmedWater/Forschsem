<# 
pipeline_yolo8.ps1 — YOLOv8 (Seg) Setup & Run, Windows / PowerShell

Beispiel:
  # CPU erzwingen und Training starten:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\pipeline_yolo8.ps1 `
    -cpu `
    -TrainScript "C:\Pfad\zu\train_yolov8_seg.py" `
    -TrainArgs "--yolo_root C:\Daten\converted\yolo --epochs 100 --imgsz 640"

  # Automatische GPU/CPU-Erkennung, env ggf. neu erzeugen:
  powershell -NoProfile -ExecutionPolicy Bypass -File .\pipeline_yolo8.ps1 -force
#>

[CmdletBinding()]
param(
  [switch]$force,
  [switch]$help,
  [switch]$cpu,                              # CPU erzwingen (sonst Auto-Erkennung)
  [string]$EnvName = "yolo8-seg",
  [string]$PythonVersion = "3.10",
  [string]$TrainScript = "",                 # Pfad zu deinem Trainingsskript (optional)
  [string]$TrainArgs   = ""                  # z.B. "--yolo_root D:\converted\yolo --epochs 100"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ----------------------------- Hilfe -----------------------------
if ($help) {
  Write-Host "Usage: .\pipeline_yolo8.ps1 [-force] [-cpu] [-EnvName name] [-PythonVersion 3.10] [-TrainScript path] [-TrainArgs args]"
  Write-Host "Creates/reuses conda env '$EnvName', installs PyTorch (+CUDA wenn vorhanden), Ultralytics & Co, optional runs your training."
  exit 0
}

Write-Host "[YOLOv8] Setup start  | Env=$EnvName  Python=$PythonVersion  Force=$($force.IsPresent)  CPU=$($cpu.IsPresent)" -ForegroundColor Cyan

# ----------------------------- Conda check -----------------------------
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
  Write-Error "❌ 'conda' nicht gefunden. Bitte Miniforge/Anaconda Prompt nutzen oder 'conda init' ausführen."
  exit 1
}

function Invoke-CondaRun {
  param([string]$Env,[string]$Cmd,[switch]$Quiet)
  if ($Quiet) {
    conda run -n $Env $Cmd | Out-Null
  } else {
    conda run -n $Env $Cmd
  }
}

# ----------------------------- Env anlegen/weiterverwenden -----------------------------
function New-Or-ReuseEnv {
  param([string]$Name,[string]$PyVer,[bool]$Force)
  $exists = conda env list | Select-String "^\s*$([Regex]::Escape($Name))\s"
  if ($exists) {
    if ($Force) {
      Write-Host "[Env] Entferne bestehende Env '$Name' (force)..." -ForegroundColor Yellow
      conda env remove -n $Name -y | Out-Null
    } else {
      Write-Host "[Env] Reuse bestehende Env '$Name'." -ForegroundColor Green
      return
    }
  }
  Write-Host "[Env] Erzeuge Env '$Name' (Python $PyVer)..." -ForegroundColor Cyan
  conda create -n $Name -c conda-forge python=$PyVer -y | Out-Null
}

# ----------------------------- Install PyTorch (GPU/CPU) -----------------------------
function Install-PyTorchStack {
  param([string]$Name,[switch]$UseCPU)
  if ($UseCPU) {
    Write-Host "[PyTorch] Installiere CPU-Stack..." -ForegroundColor Cyan
    conda install -n $Name -y -c pytorch `
      pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 cpuonly | Out-Null
  } else {
    Write-Host "[PyTorch] Installiere CUDA-Stack (12.1)..." -ForegroundColor Cyan
    conda install -n $Name -y -c pytorch -c nvidia `
      pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=12.1 | Out-Null
  }
}

# ----------------------------- Base-Stack -----------------------------
function Install-BaseStack {
  param([string]$Name)
  Write-Host "[Base] Installiere Sci-Stack (numpy<2, pandas, opencv, pillow, requests, pyyaml)..." -ForegroundColor Cyan
  conda install -n $Name -y -c conda-forge `
    "numpy<2" pandas pyyaml opencv pillow requests | Out-Null
}

# ----------------------------- Pip-Pakete -----------------------------
function Install-PipPkgs {
  param([string]$Name,[switch]$GPU)
  Write-Host "[Pip] Installiere Ultralytics & ONNX-Stack..." -ForegroundColor Cyan
  # onnxruntime CPU ist universell; onnxruntime-gpu könntest du bei Bedarf ergänzen.
  Invoke-CondaRun -Env $Name -Cmd "python -m pip install --no-cache-dir -U `
    ultralytics==8.3.223 `
    pycocotools>=2.0.0,<3.0.0 `
    onnx==1.19.1 `
    onnxruntime-gpu==1.16.3 `
    onnxslim==0.1.72 `
    albumentations==1.4.24 `
    thop==0.1.1.post2209072238" -Quiet
}

# ----------------------------- Install/Verify Pipeline -----------------------------
New-Or-ReuseEnv -Name $EnvName -PyVer $PythonVersion -Force:$force

# GPU/CPU Entscheidung
$useCPU = $false
if ($cpu) { $useCPU = $true }
else {
  # schnelle Vorprüfung: nvidia-smi vorhanden?
  $nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if (-not $nvsmi) { $useCPU = $true }
}

Install-PyTorchStack -Name $EnvName -UseCPU:$useCPU
Install-BaseStack    -Name $EnvName
Install-PipPkgs      -Name $EnvName -GPU:(!$useCPU)

# ----------------------------- Self-Test -----------------------------
Write-Host "[Check] Prüfe Installation & CUDA-Fähigkeit..." -ForegroundColor Cyan
$check = @"
import sys
out = {}
try:
  import torch
  out["torch"] = torch.__version__
  out["cuda_available"] = torch.cuda.is_available()
  out["cuda_device"] = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
except Exception as e:
  out["torch_error"] = str(e)
try:
  import ultralytics
  out["ultralytics"] = ultralytics.__version__
except Exception as e:
  out["ultralytics_error"] = str(e)
try:
  import onnx, onnxruntime as ort
  out["onnx"] = onnx.__version__
  out["onnxruntime"] = ort.__version__
  out["ort_providers"] = ort.get_available_providers()
except Exception as e:
  out["onnx_error"] = str(e)
print(out)
"@
Invoke-CondaRun -Env $EnvName -Cmd "python - <<'PY'
$($check)
PY"

# ----------------------------- Optional: Training starten -----------------------------
if ($TrainScript -ne "") {
  if (-not (Test-Path -LiteralPath $TrainScript)) {
    Write-Error "TrainScript nicht gefunden: $TrainScript"
    exit 1
  }
  Write-Host "[Run] Starte Training: $TrainScript $TrainArgs" -ForegroundColor Green
  Invoke-CondaRun -Env $EnvName -Cmd "python `"$TrainScript`" $TrainArgs"
} else {
  Write-Host "[Done] Env '$EnvName' ist bereit. Aktiviere mit: 'conda activate $EnvName'." -ForegroundColor Green
  Write-Host "       Zum Starten eines Trainings:  python Pfad\zu\train_yolov8_seg.py --yolo_root <dir> --epochs 100 --imgsz 640"
}
