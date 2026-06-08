param(
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 18000,
    [string]$UpstreamBaseUrl = "http://127.0.0.1:8000",
    [string]$UpstreamModel = "vl-model",
    [string]$DebugDir = ""
)

$ErrorActionPreference = "Stop"

$env:UPSTREAM_BASE_URL = $UpstreamBaseUrl
$env:UPSTREAM_MODEL = $UpstreamModel
if ($DebugDir) {
    $env:ADAPTER_DEBUG_DIR = $DebugDir
}

python -m uvicorn mineru_adapter.api:app --host $HostAddress --port $Port
