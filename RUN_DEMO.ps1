param([string]$Python = '', [switch]$Report)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $localVenv = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    $testedPython = Join-Path $env:USERPROFILE 'miniconda3\envs\diffusion-esi\python.exe'
    if (Test-Path -LiteralPath $localVenv) { $Python = $localVenv }
    elseif (Test-Path -LiteralPath $testedPython) { $Python = $testedPython }
    else { $Python = 'python' }
}
$oldUtf8 = $env:PYTHONUTF8
$oldBytecode = $env:PYTHONDONTWRITEBYTECODE
try {
    $env:PYTHONUTF8 = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $entry = if ($Report) { 'scripts\run_report.py' } else { 'run_demo.py' }
    & $Python (Join-Path $PSScriptRoot $entry)
    if ($LASTEXITCODE -ne 0) { throw 'KGRL demo failed; inspect the error above.' }
} finally {
    $env:PYTHONUTF8 = $oldUtf8
    $env:PYTHONDONTWRITEBYTECODE = $oldBytecode
}
