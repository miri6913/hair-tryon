# Runs the LiteRT benchmark app on one device, one (model, config) per process launch.
# Usage: powershell -File scripts\run_litert_bench.ps1 -Serial <adb-serial> -Label s20
param(
    [Parameter(Mandatory = $true)][string]$Serial,
    [Parameter(Mandatory = $true)][string]$Label,
    [string[]]$Models = @("hair", "multi"),
    [string[]]$Configs = @("cpu", "gpu16", "gpu32"),
    [int]$TimeoutSec = 120
)
$adb = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
$root = Split-Path $PSScriptRoot -Parent
$apk = "$root\android\litert-bench\app\build\outputs\apk\debug\app-debug.apk"
$out = "$root\results\fps"
$pkg = "com.vto.litertbench"
New-Item -ItemType Directory -Force $out | Out-Null

& $adb -s $Serial install -r $apk | Select-Object -Last 1
$summary = @()
foreach ($m in $Models) {
    foreach ($c in $Configs) {
        & $adb -s $Serial shell am force-stop $pkg
        & $adb -s $Serial logcat -c
        & $adb -s $Serial shell am start -n "$pkg/.MainActivity" --es model $m --es config $c | Out-Null
        $deadline = (Get-Date).AddSeconds($TimeoutSec)
        $status = "timeout"
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 2
            $log = & $adb -s $Serial logcat -d -s VTO_BENCH
            if ($log -match "RESULT") { $status = "ok"; break }
            if ($log -match "FAILED") { $status = "failed"; break }
            if (-not (& $adb -s $Serial shell pidof $pkg)) { $status = "crashed"; break }
        }
        $lines = & $adb -s $Serial logcat -d -s VTO_BENCH
        $abort = & $adb -s $Serial logcat -d | Select-String "Abort message|Fatal signal" | ForEach-Object { $_.Line -replace '^.*?(Abort message|Fatal signal)', '$1' }
        $file = "$out\${Label}_litert_${m}_${c}.log"
        @("# status=$status") + $lines + $abort | Set-Content $file
        $res = ($lines | Where-Object { $_ -match "RESULT|FAILED" } | Select-Object -Last 1)
        Write-Host ("[{0}/{1}] {2}  {3}" -f $m, $c, $status, ($res -replace '^.*VTO_BENCH\s*:\s*', ''))
        if ($abort) { Write-Host ("    " + ($abort | Select-Object -First 2 -Unique)) }
    }
}
& $adb -s $Serial shell am force-stop $pkg
