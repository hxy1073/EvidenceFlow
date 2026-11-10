$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$record = Join-Path $root 'processes.json'
if (-not (Test-Path -LiteralPath $record)) { Write-Host '没有此文件夹启动的服务记录'; exit 0 }
$processes = Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
foreach ($name in @('web', 'rag')) {
    $entry = $processes.$name
    $process = Get-Process -Id $entry.id -ErrorAction SilentlyContinue
    if (-not $process) { continue }
    if ($process.StartTime.ToUniversalTime().Ticks -ne $entry.start_ticks -or
        $process.Path -ne $entry.path) {
        throw "$name 进程身份与启动记录不符，未停止它。"
    }
    Stop-Process -Id $entry.id
    Write-Host "已停止 $name 服务"
}
Remove-Item -LiteralPath $record
