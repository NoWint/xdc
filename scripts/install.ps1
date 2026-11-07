$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = $env:PYTHON
if (-not $python) { $python = "python" }
try {
  & $python --version
} catch {
  Write-Error "Python 3.10+ is required. Install from https://www.python.org/downloads/windows/ and enable 'Add python.exe to PATH'."
  exit 1
}
& $python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if ($LASTEXITCODE -ne 0) { Write-Error "XDC requires Python 3.10 or newer"; exit 1 }

if (-not (Test-Path ".venv")) {
  & $python -m venv .venv
}
$venvPy = Join-Path (Get-Location) ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip wheel
& $venvPy -m pip install -e ".[dev]"
& $venvPy -c "from xdc import native; print('native:', native.VERSION)"
& $venvPy -m xdc doctor

Write-Host ""
Write-Host "Installed. Next:"
Write-Host "  .\.venv\Scripts\xdc.exe init --dir .\xdc-demo --host 0.0.0.0"
Write-Host "  .\.venv\Scripts\xdc.exe coordinator --config .\xdc-demo\coordinator.yaml"
Write-Host "  .\.venv\Scripts\xdc.exe worker --config .\xdc-demo\worker1.yaml"
