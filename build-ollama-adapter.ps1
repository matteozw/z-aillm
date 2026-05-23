param(
    [ValidateSet("llama3.1-8b", "llama3.1-70b", "mixtral-8x7b")]
    [string]$Profile = "llama3.1-8b",
    [string]$ModelName = "z-aillm-tuned"
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Root = Split-Path -Parent $PSScriptRoot
    $Python = Join-Path $Root "venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) {
        $Python = "python"
    }
}

$Config = Join-Path $PSScriptRoot "training\configs\$Profile-lora.yaml"
$Modelfile = & $Python "$PSScriptRoot\tools\make_ollama_modelfile.py" --config $Config
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

ollama create $ModelName -f $Modelfile.Trim()
