# Если нужен запуск от администратора
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Start-Process powershell.exe `
        -Verb RunAs `
        -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}

$projectPath  = "C:\Users\komputer\Desktop\projects\gta5\"
$activatePath = Join-Path $projectPath "venv\Scripts\Activate.ps1"

Set-Location $projectPath

if (-not (Test-Path $activatePath)) {
    Write-Host "Не найден файл активации: $activatePath"
    exit 1
}

& $activatePath
python .\runner.py