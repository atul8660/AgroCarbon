$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$python = Get-Command python -ErrorAction SilentlyContinue
$flutter = Get-Command flutter -ErrorAction SilentlyContinue

if (-not $python) {
    throw "Python 3.11 or newer is required. Install Python and add it to PATH."
}
if (-not $flutter) {
    throw "Flutter stable is required. Install Flutter and add it to PATH."
}

$pythonVersion = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($LASTEXITCODE -ne 0) {
    throw "Could not determine the installed Python version."
}
$versionParts = $pythonVersion.Split(".")
if ([int]$versionParts[0] -lt 3 -or ([int]$versionParts[0] -eq 3 -and [int]$versionParts[1] -lt 11)) {
    throw "Python 3.11 or newer is required; found Python $pythonVersion."
}

Set-Location -LiteralPath $root
git lfs install --local
if ($LASTEXITCODE -ne 0) {
    throw "Git LFS setup failed. Install Git LFS and retry."
}
git lfs pull
if ($LASTEXITCODE -ne 0) {
    throw "Model download failed. Check Git LFS installation, network access, and repository access."
}

$requiredCheckpoints = @(
    "module1_species_classification\tree_species_classifier\outputs_pilot\checkpoints\best.pt",
    "module2_health_classification\tree_health_classifier\outputs\checkpoints\best.pt",
    "module3_tree_detection\tree_detection\outputs\checkpoints\best.pt",
    "module4_tree_segmentation\tree_segmentation\outputs\checkpoints\best_fp16.pt"
)
$missingCheckpoints = @($requiredCheckpoints | Where-Object {
    -not (Test-Path (Join-Path $root $_))
})
if ($missingCheckpoints.Count -gt 0) {
    throw "Required model checkpoints are missing: $($missingCheckpoints -join ', ')"
}

$venvPython = Join-Path $root "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    & python -m venv (Join-Path $root "backend\.venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the backend virtual environment."
    }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Could not upgrade pip in the backend virtual environment."
}
& $venvPython -m pip install -r (Join-Path $root "backend\requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Backend dependency installation failed."
}

$envPath = Join-Path $root "backend\.env"
if (-not (Test-Path $envPath)) {
    $envExample = Join-Path $root "backend\.env.example"
    $jwtSecret = & $venvPython -c "import secrets; print(secrets.token_urlsafe(48))"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not generate a local JWT secret."
    }
    $envContents = [System.IO.File]::ReadAllText($envExample)
    $envContents = [regex]::Replace(
        $envContents,
        "(?m)^JWT_SECRET_KEY=.*$",
        "JWT_SECRET_KEY=`"$jwtSecret`""
    )
    [System.IO.File]::WriteAllText(
        $envPath,
        $envContents,
        [System.Text.UTF8Encoding]::new($false)
    )
}

Set-Location -LiteralPath (Join-Path $root "agrocarbon_app")
flutter pub get
if ($LASTEXITCODE -ne 0) {
    throw "Flutter dependency installation failed."
}

Write-Host ""
Write-Host "Setup complete. Start the backend and Flutter app using the commands in README.md." -ForegroundColor Green
