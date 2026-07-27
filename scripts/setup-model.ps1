param(
    [string]$Model = "nyralabs/CrisperWhisper2.0_large",
    [string]$Output = "models/ggml-crisperwhisper-large-f16.bin",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $projectRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
    py -3 -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python 3 conversion environment."
    }
}

& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $projectRoot "requirements-convert.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Could not install model-conversion dependencies."
}

$arguments = @(
    (Join-Path $projectRoot "tools\convert_hf_to_ggml.py"),
    "--model", $Model,
    "--output", (Join-Path $projectRoot $Output),
    "--type", "f16"
)
if ($Force) {
    $arguments += "--force"
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Model conversion failed."
}
