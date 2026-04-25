Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step($message) {
    Write-Host "[STEP] $message" -ForegroundColor Cyan
}

function Write-Pass($message) {
    Write-Host "[PASS] $message" -ForegroundColor Green
}

function Fail-Fast($message) {
    Write-Host "[FAIL] $message" -ForegroundColor Red
    exit 1
}

function Resolve-PythonExe {
    function Test-PythonCandidate {
        param([string]$Candidate)
        try {
            if ($Candidate -like "* -3") {
                & py -3 -c "import sys; print(sys.version)"
            } else {
                & $Candidate -c "import sys; print(sys.version)"
            }
            return ($LASTEXITCODE -eq 0)
        } catch {
            return $false
        }
    }

    $candidates = @()

    if ($env:SSS_PYTHON -and (Test-Path $env:SSS_PYTHON)) {
        $candidates += $env:SSS_PYTHON
    }

    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $candidates += $venvPython
    }

    $parentVenvPython = Join-Path (Join-Path $PSScriptRoot "..") ".venv\Scripts\python.exe"
    if (Test-Path $parentVenvPython) {
        $candidates += (Resolve-Path $parentVenvPython).Path
    }

    $pgAdminPython = "C:\Program Files\PostgreSQL\17\pgAdmin 4\python\python.exe"
    if (Test-Path $pgAdminPython) {
        $candidates += $pgAdminPython
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $candidates += $pythonCmd.Source
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $candidates += "$($pyCmd.Source) -3"
    }

    foreach ($candidate in $candidates | Select-Object -Unique) {
        # Skip common non-functional Windows Store shim paths.
        if ($candidate -like "*\WindowsApps\python.exe") {
            continue
        }
        if (Test-PythonCandidate -Candidate $candidate) {
            return $candidate
        }
    }

    return $null
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Code
    )

    Write-Step $Name

    $bootstrap = @"
import os
import site
import sys

repo_root = r"$($PSScriptRoot)"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    user_site = site.getusersitepackages()
except Exception:
    user_site = None

if user_site and os.path.isdir(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)
"@

    $tmpFile = Join-Path $env:TEMP ("sss_demo_runner_" + [guid]::NewGuid().ToString("N") + ".py")
    Set-Content -Path $tmpFile -Value ($bootstrap + "`r`n" + $Code) -Encoding UTF8

    try {
        if ($script:PythonExe -like "* -3") {
            & py -3 $tmpFile
        } else {
            & $script:PythonExe $tmpFile
        }
        if ($LASTEXITCODE -ne 0) {
            Fail-Fast "$Name failed (exit code $LASTEXITCODE)."
        }
    } finally {
        Remove-Item $tmpFile -Force -ErrorAction SilentlyContinue
    }

    Write-Pass $Name
}

try {
    Write-Step "Preflight: checking required project files"
    $requiredFiles = @(
        "sss_demo.py",
        "sss_stress_debug.py",
        "sss_visualize_demo.py",
        "sss_hackathon_env.py",
        "sss_training.py",
        "sss_reward_verifier.py",
        "api.py",
        "requirements.txt"
    )

    foreach ($file in $requiredFiles) {
        $path = Join-Path $PSScriptRoot $file
        if (-not (Test-Path $path)) {
            Fail-Fast "Missing required file: $file"
        }
    }
    Write-Pass "Required files found"

    Write-Step "Preflight: resolving Python interpreter"
    $script:PythonExe = Resolve-PythonExe
    if (-not $script:PythonExe) {
        Fail-Fast "No Python interpreter found. Set SSS_PYTHON or create .venv."
    }
    Write-Pass "Using Python: $script:PythonExe"

    Invoke-PythonStep -Name "Preflight: import dependency checks" -Code @'
import importlib.util
import json
import sys

required = ["fastapi", "matplotlib", "pytest"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {missing}. Install with requirements.txt.")

print(json.dumps({"dependency_check": "ok", "python": sys.executable}, indent=2))
'@

    Invoke-PythonStep -Name "Judge Flow 1/4: generate baseline-vs-trained demo outputs" -Code @'
import json
from pathlib import Path
import sss_demo

results = sss_demo.run_demo()
out = {
    "improvement": results["improvement"],
    "standard_baseline_reward": results["baseline_metrics"]["avg_total_reward"],
    "standard_trained_reward": results["trained_metrics"]["avg_total_reward"],
}
print(json.dumps(out, indent=2))

expected = Path("demo_outputs/demo_results.json")
if not expected.exists():
    raise SystemExit("demo_results.json was not generated.")
'@

    Invoke-PythonStep -Name "Judge Flow 2/4: build visualization plots from demo_outputs" -Code @'
import json
from pathlib import Path
import sss_visualize_demo

sss_visualize_demo.validate_input()
plot_path = sss_visualize_demo.build_plots()
print(json.dumps({"plot_path": str(plot_path)}, indent=2))

if not Path(plot_path).exists():
    raise SystemExit("Visualization plot file was not created.")
'@

    Invoke-PythonStep -Name "Judge Flow 3/4: run stress-debug anti-hacking checks" -Code @'
import json
import sss_stress_debug

report = sss_stress_debug.run_stress_debug()
print(json.dumps({
    "all_assertions_passed": report["all_assertions_passed"],
    "assertions": report["assertions"],
    "reward_alignment_corr": report["reward_alignment_corr"]
}, indent=2))

if not report["all_assertions_passed"]:
    raise SystemExit("Stress-debug assertions failed.")
'@

    Invoke-PythonStep -Name "Judge Flow 4/4: API smoke test (root/reset/step/grader/baseline)" -Code @'
import json
from fastapi.testclient import TestClient
import api

client = TestClient(api.app)

checks = {
    "root": client.get("/").status_code,
    "reset": client.post("/reset", json={"seed": 42}).status_code,
    "step": client.post("/step", json={"action": "increase_marketing"}).status_code,
    "grader": client.get("/grader", params={"task_name": "survival"}).status_code,
    "baseline": client.get("/baseline", params={"seed": 42}).status_code,
    "invalid_step_hire": client.post("/step", json={"action": "hire"}).status_code,
}
print(json.dumps(checks, indent=2))

required_200 = ["root", "reset", "step", "grader", "baseline"]
for key in required_200:
    if checks[key] != 200:
        raise SystemExit(f"API check failed: {key} returned {checks[key]} (expected 200).")
'@

    Invoke-PythonStep -Name "Final consistency summary" -Code @'
import json
from pathlib import Path

data = json.loads(Path("demo_outputs/demo_results.json").read_text(encoding="utf-8"))
summary = {
    "standard_improvement": data["improvement"],
    "recession_improvement": data["scenario_results"]["recession"]["improvement"],
    "competition_improvement": data["scenario_results"]["competition"]["improvement"],
    "artifact_files": [
        "demo_outputs/demo_results.json",
        "demo_outputs/policy_comparison_plots.png",
        "demo_outputs/trained_policy_qtable.json",
    ],
}
print(json.dumps(summary, indent=2))
'@

    Write-Pass "Full hackathon judge flow completed successfully."
    exit 0
} catch {
    Fail-Fast $_.Exception.Message
}
