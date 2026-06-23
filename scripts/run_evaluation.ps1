# Activate venv and evaluate both sample videos
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Activate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) {
    Write-Host "Virtual environment not found. Run: python -m venv .venv"
    exit 1
}

& $Activate
Write-Host "Virtual environment activated."
Write-Host ""

python scripts/evaluate_videos.py @args
