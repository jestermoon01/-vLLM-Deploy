# vLLM Deployment

Start the deployment from PowerShell in this folder:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\run_deploy.ps1"
```

The script starts or reuses vLLM, checks inference, and prints the chat UI
address. The UI can be exposed to devices on the local subnet; it has no login.

Stop the project and remove its managed LAN access:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\stop_deploy.ps1"
```

See [docs.md](./docs.md) for setup, options, security notes, and troubleshooting.
