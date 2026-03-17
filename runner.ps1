$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Start-Process powershell.exe `
        -Verb RunAs `
        -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}
$projectPath = "C:\Path\To\Your\Project"
$activatePath = ".\venv\Scripts\Activate.ps1"

Set-Location $projectPath
& $activatePath
python .\runner.py