
<# 
run_detectron2_docker.ps1 — Detectron2 in Docker unter Windows starten

Nutzung:
  # CPU:
  .\run_detectron2_docker.ps1
  .\run_detectron2_docker.ps1 -Mode cpu

  # GPU (WSL2 + NVIDIA Container Toolkit erforderlich):
  .\run_detectron2_docker.ps1 -Mode gpu
#>

param(
  [ValidateSet("cpu","gpu")]
  [string]$Mode = "cpu"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Pfade bestimmen
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$DockerDir = $ScriptDir
$ProjectRoot = Resolve-Path (Join-Path $DockerDir "..\..")

Write-Host "[DOCKER] Building Docker image from: $DockerDir"
Write-Host "[DOCKER] Project root: $ProjectRoot"

# Prüfungen
if (-not (Test-Path (Join-Path $DockerDir "Dockerfile"))) {
  Write-Error "[ERROR] Dockerfile not found at: $(Join-Path $DockerDir "Dockerfile")"
}

if (-not (Test-Path (Join-Path $DockerDir "train_detectron2.py"))) {
  Write-Error "[ERROR] Training script not found at: $(Join-Path $DockerDir "train_detectron2.py")"
}

# Image bauen
Write-Host "[DOCKER] Building Docker image (this may take a few minutes)..."
Push-Location $DockerDir
docker build -t detectron2-training .
Pop-Location

# Windows-Pfad für Docker sauber machen (Backslashes -> Slashes)
$ProjectRootForDocker = ($ProjectRoot.Path -replace '\\','/')

if ($Mode -eq "gpu") {
  Write-Host "[DOCKER] Running with GPU support..."
  docker run --gpus all -it `
    -v "$ProjectRootForDocker:/workspace" `
    -w /workspace `
    -e DETECTRON2_USE_CUDA=1 `
    detectron2-training `
    python docker/detectron2/train_detectron2.py
}
else {
  Write-Host "[DOCKER] Running in CPU mode..."
  docker run -it `
    -v "$ProjectRootForDocker:/workspace" `
    -w /workspace `
    -e DETECTRON2_USE_CUDA=0 `
    detectron2-training `
    python docker/detectron2/train_detectron2.py
}
``
