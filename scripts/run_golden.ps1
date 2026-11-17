# Golden test driver: push a photo, render it with the app's still-image mode (Refine off), pull the result and
# compare against the Python reference (scripts/compare_golden.py).
#   .\scripts\run_golden.ps1 -Photos kaggle_test_18-20_25,kaggle_test_18-20_30
# Needs the debug APK installed (android/hairtryon: .\gradlew.bat assembleDebug; adb install -r ...).
param(
    [string[]]$Photos = @("kaggle_test_18-20_25"),
    [string]$Device = "",
    [int]$WaitSeconds = 6,
    [string]$Shade = "honey",   # palette key, see Palette.kt / SHADES in recolor2.py
    [string]$Python = "$env:USERPROFILE\anaconda3\envs\vto\python.exe"   # env `vto`, called directly: `conda run` swallows output
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$adb = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
$sel = if ($Device) { @("-s", $Device) } else { @() }   # no -Device: adb uses the only connected device
$pkg = "com.vto.hairtryon"
$dir = "/sdcard/Android/data/$pkg/files"
$outDir = Join-Path $root "results\golden"
New-Item -ItemType Directory -Force $outDir | Out-Null

$failed = 0
foreach ($name in $Photos) {
    $photo = Join-Path $root "data\photos_kaggle\$name.jpg"
    if (-not (Test-Path $photo)) { $photo = Join-Path $root "data\photos_wiki\$name.jpg" }
    if (-not (Test-Path $photo)) { throw "photo not found: $name" }

    & $adb @sel shell am force-stop $pkg
    & $adb @sel shell mkdir -p $dir
    & $adb @sel shell rm -f "$dir/still_out.png" "$dir/still_mask.f32"
    & $adb @sel push $photo "$dir/still.jpg" | Out-Null
    & $adb @sel shell am start -n "$pkg/.MainActivity" --es still "$dir/still.jpg" --ez refine false --es shade $Shade | Out-Null
    Start-Sleep -Seconds $WaitSeconds   # mask smoothing and mean-base estimate converge on a static frame

    # Capture is the first button of its row; find it from the UI tree instead of hard-coding pixels.
    & $adb @sel shell uiautomator dump /sdcard/ui.xml | Out-Null
    $xml = (& $adb @sel shell cat /sdcard/ui.xml) -join "`n"
    if ($xml -notmatch 'text="Capture"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"') { throw "Capture button not found (carrier dialog covering it?)" }
    $x = [int](([int]$Matches[1] + [int]$Matches[3]) / 2)
    $y = [int](([int]$Matches[2] + [int]$Matches[4]) / 2)
    & $adb @sel shell input tap $x $y
    Start-Sleep -Seconds 3

    $png = Join-Path $outDir "$name.out.png"
    $mask = Join-Path $outDir "$name.mask.f32"
    & $adb @sel pull "$dir/still_out.png" $png | Out-Null
    & $adb @sel pull "$dir/still_mask.f32" $mask | Out-Null

    & $Python (Join-Path $PSScriptRoot "compare_golden.py") --photo $photo --out $png --mask $mask --shade $Shade --save (Join-Path $outDir "$name.$Shade.compare.png")
    if ($LASTEXITCODE -ne 0) { $failed++ }
}
& $adb @sel shell am force-stop $pkg
if ($failed -gt 0) { Write-Host "$failed photo(s) failed"; exit 1 }
Write-Host "all golden comparisons passed"
