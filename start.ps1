param([switch]$NoBrowser, [int]$RagPort = 8021, [int]$WebPort = 8031)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$runtime = Join-Path $root 'runtime\python.exe'
$venv = Join-Path $root '.venv\Scripts\python.exe'
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

function Find-Python {
    if ($env:EVIDENCEFLOW_PYTHON -and (Test-Path -LiteralPath $env:EVIDENCEFLOW_PYTHON)) {
        return $env:EVIDENCEFLOW_PYTHON
    }
    if (Test-Path -LiteralPath $runtime) { return $runtime }
    if (Test-Path -LiteralPath $venv) { return $venv }
    foreach ($version in @('3.11', '3.12')) {
        try {
            $path = & py "-$version" -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $path -and (Test-Path -LiteralPath $path)) { return $path.Trim() }
        } catch {}
    }
    foreach ($name in @('python3.11', 'python3.12', 'python')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $command) { continue }
        try {
            $version = & $command.Source -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>$null
            if ($LASTEXITCODE -eq 0 -and $version -in @('3.11', '3.12')) { return $command.Source }
        } catch {}
    }
    throw '需要 Windows 64 位 Python 3.11 或 3.12。安装后重新双击 start.bat。'
}

$python = Find-Python
if ($python -ne $runtime -and $python -ne $venv -and -not $env:EVIDENCEFLOW_PYTHON) {
    Write-Host '首次运行：建立独立 Python 环境并安装依赖…'
    & $python -m venv (Join-Path $root '.venv')
    if ($LASTEXITCODE -ne 0) { throw '创建 Python 环境失败' }
    $python = $venv
    & $python -m pip install -r (Join-Path $root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw '依赖安装失败，请检查网络连接' }
}

foreach ($port in @($RagPort, $WebPort)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        throw "端口 $port 已被占用。请先关闭原服务或运行 stop.ps1。"
    }
}

$env:EVIDENCEFLOW_HOST = '127.0.0.1'
$env:EVIDENCEFLOW_PORT = [string]$WebPort
$env:EVIDENCEFLOW_RAG_HOST = '127.0.0.1'
$env:EVIDENCEFLOW_RAG_PORT = [string]$RagPort
$env:EVIDENCEFLOW_RAG_URL = "http://127.0.0.1:$RagPort"
$ragDir = Join-Path $root 'rag-service'
$webDir = Join-Path $root 'evidence-flow'
$rag = Start-Process -FilePath $python -ArgumentList 'serve_h_faiss.py' -WorkingDirectory $ragDir -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'rag.out.log') -RedirectStandardError (Join-Path $logDir 'rag.err.log')
Write-Host '正在启动文献检索服务（初次建立缓存可能需要约一分钟）…'
$ragReady = $false
for ($i = 0; $i -lt 180; $i++) {
    if ($rag.HasExited) { break }
    try {
        $health = Invoke-RestMethod "http://127.0.0.1:$RagPort/health" -TimeoutSec 2
        if ($health.status -eq 'healthy') { $ragReady = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}
if (-not $ragReady) {
    Write-Host '检索服务启动失败，错误日志：' -ForegroundColor Red
    Get-Content (Join-Path $logDir 'rag.err.log') -Tail 30 -ErrorAction SilentlyContinue
    throw "无法启动 $RagPort 检索服务"
}

$web = Start-Process -FilePath $python -ArgumentList 'app.py' -WorkingDirectory $webDir -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'web.out.log') -RedirectStandardError (Join-Path $logDir 'web.err.log')
$webReady = $false
for ($i = 0; $i -lt 30; $i++) {
    if ($web.HasExited) { break }
    try {
        $response = Invoke-WebRequest "http://127.0.0.1:$WebPort/" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) { $webReady = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}
if (-not $webReady) {
    Stop-Process -Id $rag.Id -ErrorAction SilentlyContinue
    Write-Host '网页服务启动失败，错误日志：' -ForegroundColor Red
    Get-Content (Join-Path $logDir 'web.err.log') -Tail 30 -ErrorAction SilentlyContinue
    throw "无法启动 $WebPort 网页服务"
}

@{ rag = @{ id = $rag.Id; start_ticks = $rag.StartTime.ToUniversalTime().Ticks; path = $python }; web = @{ id = $web.Id; start_ticks = $web.StartTime.ToUniversalTime().Ticks; path = $python } } |
    ConvertTo-Json -Depth 3 | Set-Content -Path (Join-Path $root 'processes.json') -Encoding UTF8
Write-Host "EvidenceFlow 已启动：http://127.0.0.1:$WebPort/" -ForegroundColor Green
if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$WebPort/" }
