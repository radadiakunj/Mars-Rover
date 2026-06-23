# Install CUDA-enabled PyTorch into the project venv (RTX 4060 / CUDA 12.x)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

& "$ProjectRoot\.venv\Scripts\Activate.ps1"

Write-Host "Installing CUDA PyTorch (cu124 wheel) ..."
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host ""
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
