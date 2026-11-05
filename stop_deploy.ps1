[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$Distribution = "Ubuntu",
    [switch]$RemoveLanRule
)

$ErrorActionPreference = "Stop"
$ruleName = "vLLM chat UI LAN (local subnet only)"

function Remove-ProjectLanAccess {
    $rule = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    if (-not $rule) {
        Write-Host "No managed LAN firewall rule found; leaving unowned port-forward rules unchanged."
        return
    }

    $principal = New-Object Security.Principal.WindowsPrincipal(
        [Security.Principal.WindowsIdentity]::GetCurrent()
    )
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Removing the project's LAN access requires Administrator privileges."
    }

    $localAddresses = $rule | Get-NetFirewallAddressFilter |
        Select-Object -ExpandProperty LocalAddress
    foreach ($address in $localAddresses) {
        if ($address -and $address -ne "Any") {
            & netsh.exe interface portproxy delete v4tov4 `
                "listenaddress=$address" listenport=8000 protocol=tcp | Out-Null
            $listener = Get-NetTCPConnection `
                -State Listen `
                -LocalAddress $address `
                -LocalPort 8000 `
                -ErrorAction SilentlyContinue
            if ($listener) {
                throw "A listener remains on $address`:8000; leaving the firewall rule in place."
            }
        }
    }
    $rule | Remove-NetFirewallRule
    Write-Host "Removed the project's LAN firewall rule and its port 8000 forward."
}

if ($RemoveLanRule) {
    Remove-ProjectLanAccess
    exit 0
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "wsl.exe was not found."
}

$linuxPath = (& wsl.exe --distribution $Distribution --exec wslpath -a $PSScriptRoot | Out-String).Trim()
if ($LASTEXITCode -ne 0 -or -not $linuxPath) {
    throw "Could not access this folder from WSL distribution '$Distribution'."
}

$quotedPath = "'" + $linuxPath.Replace("'", "'\''") + "'"
$dryRunArgument = if ($WhatIfPreference) { " --dry-run" } else { "" }
$linuxScript = @'
set -e
source_path=__SOURCE__
stage_path="$HOME/vllm-deploy"
if [ ! -x "$stage_path/venv/bin/python" ]; then
    echo "The staged vLLM Python environment was not found at $stage_path."
    echo "No WSL processes were stopped."
    exit 2
fi
cp -f "$source_path/vllm_deploy/stop.py" "$stage_path/vllm_deploy/stop.py"
cd -- "$stage_path"
exec "$stage_path/venv/bin/python" -m vllm_deploy.stop --project-dir "$source_path"__DRY_RUN__
'@.Replace("`r`n", "`n").Replace("__SOURCE__", $quotedPath).Replace(
    "__DRY_RUN__",
    $dryRunArgument
)
$scriptBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($linuxScript))
$linuxCommand = "printf '%s' '$scriptBase64' | base64 -d | bash"
& wsl.exe --distribution $Distribution --exec bash -lc $linuxCommand
if ($LASTEXITCODE -ne 0) {
    throw "Stopping the WSL project processes failed (exit $LASTEXITCODE)."
}

if ($WhatIfPreference) {
    Write-Host "Dry run complete; no processes or Windows network rules were changed."
    return
}

$rule = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if ($rule) {
    $principal = New-Object Security.Principal.WindowsPrincipal(
        [Security.Principal.WindowsIdentity]::GetCurrent()
    )
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        $arguments = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", "`"$PSCommandPath`"",
            "-RemoveLanRule"
        )
        $elevated = Start-Process `
            -FilePath "powershell.exe" `
            -Verb RunAs `
            -Wait `
            -PassThru `
            -ArgumentList $arguments
        if ($elevated.ExitCode -ne 0) {
            throw "Removing the project's LAN firewall rule failed or was cancelled (exit $($elevated.ExitCode))."
        }
    } else {
        Remove-ProjectLanAccess
    }
} else {
    Write-Host "No managed LAN firewall rule found; leaving unowned port-forward rules unchanged."
}
