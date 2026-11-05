[CmdletBinding()]
param(
    [string]$Distribution = "Ubuntu",
    [switch]$CheckOnly,
    [switch]$StageOnly,
    [switch]$ConfigureLan,
    [string]$WslAddress,
    [string]$LanAddress
)

$ErrorActionPreference = "Stop"

if ($ConfigureLan) {
    $principal = New-Object Security.Principal.WindowsPrincipal(
        [Security.Principal.WindowsIdentity]::GetCurrent()
    )
    if (-not $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )) {
        throw "The LAN forwarding setup must run as Administrator."
    }

    & netsh.exe interface portproxy delete v4tov4 `
        "listenaddress=$LanAddress" listenport=8000 protocol=tcp | Out-Null
    & netsh.exe interface portproxy add v4tov4 `
        "listenaddress=$LanAddress" listenport=8000 `
        "connectaddress=$WslAddress" connectport=8000 protocol=tcp
    if ($LASTEXITCODE -ne 0) {
        throw "Could not forward $LanAddress`:8000 to WSL at $WslAddress`:8000."
    }

    $ruleName = "vLLM chat UI LAN (local subnet only)"
    Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue |
        Remove-NetFirewallRule
    New-NetFirewallRule `
        -DisplayName $ruleName `
        -Direction Inbound `
        -Action Allow `
        -Protocol TCP `
        -LocalPort 8000 `
        -LocalAddress $LanAddress `
        -RemoteAddress LocalSubnet `
        -Profile Any | Out-Null
    return
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "wsl.exe was not found. Install WSL and an Ubuntu distribution first."
}

$linuxPath = (& wsl.exe --distribution $Distribution --exec wslpath -a $PSScriptRoot | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $linuxPath) {
    throw "Could not access this folder from WSL distribution '$Distribution'."
}

$quotedPath = "'" + $linuxPath.Replace("'", "'\''") + "'"
if ($CheckOnly) {
    $linuxCommand = "cd -- $quotedPath && python3 --version && nvidia-smi -L"
    & wsl.exe --distribution $Distribution --exec bash -lc $linuxCommand
    exit $LASTEXITCODE
}

$linuxScript = @'
set -e
source_path=__SOURCE__
stage_path="$HOME/vllm-deploy"
mkdir -p "$stage_path/vllm_deploy"
cp -f "$source_path/deploy_vllm.py" "$source_path/test_server.py" "$source_path/requirements.lock.txt" "$stage_path/"
cp -a "$source_path/vllm_deploy/." "$stage_path/vllm_deploy/"
rm -rf "$stage_path/__pycache__" "$stage_path/vllm_deploy/__pycache__"
for name in .env .api_key; do
    if [ -f "$source_path/$name" ]; then
        cp -f "$source_path/$name" "$stage_path/$name"
        chmod 600 "$stage_path/$name"
    else
        rm -f "$stage_path/$name"
    fi
done
echo "Running from WSL-native path: $stage_path"
cd -- "$stage_path"
if [ "__STAGE_ONLY__" = "true" ]; then
    python3 --version
    nvidia-smi -L
    echo "Files staged. Run run_deploy.ps1 again without -StageOnly to start the servers."
else
    # Prefer the WSL-native staged environment, but fall back to the project's
    # existing WSL virtualenv when the staging directory has not been prepared
    # yet. This keeps a fresh run from failing with a missing venv/bin/python.
    if [ -x "$stage_path/venv/bin/python" ]; then
        python_bin="$stage_path/venv/bin/python"
    elif [ -x "$source_path/venv/bin/python" ]; then
        python_bin="$source_path/venv/bin/python"
    else
        echo "No usable WSL Python environment found. Run the deployment installer first." >&2
        exit 1
    fi
    exec "$python_bin" -m vllm_deploy.launch
fi
'@.Replace("`r`n", "`n").Replace("__SOURCE__", $quotedPath).Replace(
    "__STAGE_ONLY__",
    $(if ($StageOnly) { "true" } else { "false" })
)

$scriptBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($linuxScript))
$linuxCommand = "printf '%s' '$scriptBase64' | base64 -d | bash"
& wsl.exe --distribution $Distribution --exec bash -lc $linuxCommand
if ($LASTEXITCODE -ne 0) {
    throw "vLLM deployment failed inside WSL (exit $LASTEXITCODE)."
}

if ($StageOnly) {
    exit 0
}

$wslAddresses = (& wsl.exe --distribution $Distribution --exec hostname -I | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Could not determine the WSL IPv4 address."
}
$wslAddress = [regex]::Matches($wslAddresses, '\b(?:\d{1,3}\.){3}\d{1,3}\b') |
    ForEach-Object { $_.Value } |
    Where-Object { $_ -notlike "127.*" -and $_ -notlike "169.254.*" } |
    Select-Object -First 1
if (-not $wslAddress) {
    throw "Could not determine a usable WSL IPv4 address."
}

$defaultRoute = Get-NetRoute -DestinationPrefix "0.0.0.0/0" |
    Sort-Object RouteMetric, InterfaceMetric |
    Select-Object -First 1
$lanAddress = Get-NetIPAddress -AddressFamily IPv4 -InterfaceIndex $defaultRoute.InterfaceIndex |
    Where-Object {
        $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*"
    } |
    Select-Object -ExpandProperty IPAddress -First 1
if (-not $lanAddress) {
    throw "Could not determine the Windows LAN IPv4 address."
}

$principal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent()
)
if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    & $PSCommandPath -ConfigureLan -WslAddress $wslAddress -LanAddress $lanAddress -Distribution $Distribution
    if ($LASTEXITCODE -ne 0) {
        throw "Could not configure Windows LAN forwarding."
    }
} else {
    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$PSCommandPath`"",
        "-ConfigureLan",
        "-WslAddress", $wslAddress,
        "-LanAddress", $lanAddress,
        "-Distribution", $Distribution
    )
    $elevated = Start-Process `
        -FilePath "powershell.exe" `
        -Verb RunAs `
        -Wait `
        -PassThru `
        -ArgumentList $arguments
    if ($elevated.ExitCode -ne 0) {
        throw "Windows LAN forwarding setup failed or was cancelled (exit $($elevated.ExitCode))."
    }
}

$healthUrl = "http://${lanAddress}:8000/health"
try {
    $health = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 10
} catch {
    throw "LAN health check failed at $healthUrl`: $($_.Exception.Message)"
}
if ($health.StatusCode -ne 200) {
    throw "LAN health check returned HTTP $($health.StatusCode)."
}

Write-Host ""
Write-Host "Deployment is ready."
Write-Host "Hosted chat UI: http://${lanAddress}:8000"
Write-Host "Health check:   $healthUrl"
Write-Host "vLLM API:       http://127.0.0.1:8001 (localhost only; not exposed to the LAN)"
Write-Host "Warning: the chat UI has no login; local-subnet devices can submit prompts."
