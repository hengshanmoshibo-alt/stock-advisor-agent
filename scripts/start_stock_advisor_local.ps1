param(
    [string]$Python = "D:\home\software\anaconda3\envs\whisper-gpu\python.exe",
    [switch]$NoFrontend
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Frontend = Join-Path $Root "frontend"
$BackendOut = Join-Path $Root "stock-advisor-backend.out.log"
$BackendErr = Join-Path $Root "stock-advisor-backend.err.log"
$FrontendOut = Join-Path $Root "frontend-dev.out.log"
$FrontendErr = Join-Path $Root "frontend-dev.err.log"

function Stop-PortProcess([int]$Port) {
    $connections = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($connection in $connections) {
        $process = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
        if ($process -and ($process.ProcessName -match "python|uvicorn|node")) {
            Stop-Process -Id $process.Id -Force
        }
    }
}

function Test-Health([string]$Url, [int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    do {
        try {
            $response = Invoke-RestMethod -Uri $Url -TimeoutSec 2
            return $response
        } catch {
            Start-Sleep -Milliseconds 700
        }
    } while ((Get-Date) -lt $deadline)
    throw "Health check failed: $Url"
}

Set-Location $Root
$env:PYTHONPATH = Join-Path $Root "src"

Stop-PortProcess 8020
Start-Process -FilePath $Python `
    -ArgumentList @("scripts\run_stock_advisor_backend.py") `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $BackendOut `
    -RedirectStandardError $BackendErr `
    -WindowStyle Hidden

$health = Test-Health "http://127.0.0.1:8020/api/health" 30
Write-Host "Stock advisor backend ready: $($health.project), model=$($health.model_mode)"

if (-not $NoFrontend) {
    Stop-PortProcess 5173
    Start-Process -FilePath "npm.cmd" `
        -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1") `
        -WorkingDirectory $Frontend `
        -RedirectStandardOutput $FrontendOut `
        -RedirectStandardError $FrontendErr `
        -WindowStyle Hidden
    Start-Sleep -Seconds 2
    Write-Host "Frontend starting: http://127.0.0.1:5173/"
}
