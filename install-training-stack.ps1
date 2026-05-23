$ErrorActionPreference = "Stop"
$TrainingVenv = Join-Path $PSScriptRoot ".venv"
$Python = Join-Path $TrainingVenv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    python -m venv $TrainingVenv
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r "$PSScriptRoot\training\requirements.txt"
