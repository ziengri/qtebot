$projectPath = "C:\Path\To\Your\Project"
$activatePath = ".\venv\Scripts\Activate.ps1"

Set-Location $projectPath
& $activatePath
python .\runner.py