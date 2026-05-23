param(
    [ValidateSet("llama3.1-8b", "llama3.1-70b", "mixtral-8x7b")]
    [string]$Profile = "llama3.1-8b",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

$Config = Join-Path $PSScriptRoot "training\configs\$Profile-lora.yaml"
$ArgsList = @("$PSScriptRoot\tools\train_lora.py", "--config", $Config)
if ($DryRun) {
    $ArgsList += "--dry-run"
}

& $Python @ArgsList
