# build_and_install.ps1
# Workaround for Flutter CLI false-positive error when project path contains spaces.
# Flutter's tool cannot locate the APK after a successful Gradle build when the
# project path has spaces (e.g. "Capston Zip Files"). The APK IS built; we just
# install it with adb directly.

$APK = "android\app\build\outputs\flutter-apk\app-debug.apk"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " AgroCarbon App - Build & Install Tool  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Clean
Write-Host "[1/3] Running flutter clean..." -ForegroundColor Yellow
flutter clean
if ($LASTEXITCODE -ne 0) { Write-Host "flutter clean failed." -ForegroundColor Red; exit 1 }

# Step 2: Build (ignore the "couldn't find it" false-positive at the end)
Write-Host ""
Write-Host "[2/3] Building debug APK (the 'Gradle build failed' message at the end is a false positive)..." -ForegroundColor Yellow
flutter build apk --debug 2>&1 | Out-Null
# Check if APK actually exists regardless of flutter exit code
if (-Not (Test-Path $APK)) {
    Write-Host "ERROR: APK not found at expected path. Check Gradle output for real errors." -ForegroundColor Red
    exit 1
}
$size = (Get-Item $APK).Length
Write-Host "APK built successfully! Size: $([math]::Round($size/1MB, 1)) MB" -ForegroundColor Green

# Step 3: Install via adb
Write-Host ""
Write-Host "[3/3] Installing APK on connected device via adb..." -ForegroundColor Yellow
adb install -r $APK
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "SUCCESS! AgroCarbon app installed on device." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "adb install failed. Make sure a device is connected and USB debugging is enabled." -ForegroundColor Red
    Write-Host "You can also manually install the APK from: $APK" -ForegroundColor Yellow
}
