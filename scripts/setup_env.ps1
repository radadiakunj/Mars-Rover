# Create and activate virtual environment (Windows PowerShell)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Creating virtual environment at .venv ..."
python -m venv .venv

Write-Host "Activating virtual environment ..."
& "$ProjectRoot\.venv\Scripts\Activate.ps1"

Write-Host "Upgrading pip ..."
python -m pip install --upgrade pip

Write-Host "Installing dependencies ..."
pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete. Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
