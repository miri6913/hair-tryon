# Thermal soak: launch the app and sample battery temperature, thermal status and the app's VTO_FPS line every
# -IntervalSeconds for -DurationSeconds. Point the camera at a face (or a photo of one) with a shade selected.
#   .\scripts\run_soak.ps1                                  # S20, 5 minutes
#   .\scripts\run_soak.ps1 -Device <adb-serial> -Tag tab_a8  # Galaxy Tab A8
# Output: results/fps/soak_<tag>_<date>.csv
param(
    [string]$Device = "",
    [string]$Tag = "s20",
    [int]$DurationSeconds = 300,
    [int]$IntervalSeconds = 10,
    [string]$Shade = "honey"   # palette key
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$adb = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
$sel = if ($Device) { @("-s", $Device) } else { @() }   # no -Device: adb uses the only connected device
$pkg = "com.vto.hairtryon"
$outDir = Join-Path $root "results\fps"
New-Item -ItemType Directory -Force $outDir | Out-Null
$csv = Join-Path $outDir ("soak_{0}_{1}.csv" -f $Tag, (Get-Date -Format "yyyy-MM-dd"))

& $adb @sel shell am force-stop $pkg
& $adb @sel logcat -c
& $adb @sel shell am start -n "$pkg/.MainActivity" --es shade $Shade | Out-Null
Start-Sleep -Seconds 5

"t_s,battery_temp_c,thermal_status,render_fps,mask_lag_ms,readback_ms,edge_flicker" | Set-Content $csv -Encoding utf8
$start = Get-Date
while (((Get-Date) - $start).TotalSeconds -lt $DurationSeconds) {
    $t = [int]((Get-Date) - $start).TotalSeconds
    $bat = (& $adb @sel shell dumpsys battery) -join "`n"
    $temp = if ($bat -match 'temperature:\s*(\d+)') { [double]$Matches[1] / 10 } else { "" }
    $thermal = (& $adb @sel shell dumpsys thermalservice) -join "`n"
    $status = if ($thermal -match 'Thermal Status:\s*(\d+)') { $Matches[1] } else { "" }
    $log = (& $adb @sel logcat -d -s VTO_FPS:I) | Select-String "render_fps=" | Select-Object -Last 1   # seg_fps lines share the tag
    $fps = ""; $lag = ""; $rb = ""; $flick = ""
    if ($log -match 'render_fps=([\d.]+).*mask_lag_ms=([\d.]+) readback_ms=([\d.]+).*edge_flicker=([\d.]+)') {
        $fps = $Matches[1]; $lag = $Matches[2]; $rb = $Matches[3]; $flick = $Matches[4]
    }
    "$t,$temp,$status,$fps,$lag,$rb,$flick" | Add-Content $csv -Encoding utf8
    Write-Host "t=$t s  temp=$temp C  thermal=$status  fps=$fps  lag=$lag ms"
    Start-Sleep -Seconds $IntervalSeconds
}
& $adb @sel shell am force-stop $pkg
Write-Host "wrote $csv"
