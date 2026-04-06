"""FastAPI application for the Startup Survival Simulator."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from baseline import run_baseline
from env import StartupEnv
from grader import grade
from models import ResetRequest, StepRequest
from tasks import get_tasks

app = FastAPI(
    title="Startup Survival Simulator",
    description="OpenEnv-style startup decision environment for hackathon evaluation.",
    version="1.0.0",
)

# Keep a single in-process environment instance for the simple demo API.
env = StartupEnv(seed=42)


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Human-friendly landing page that also guarantees an HTTP 200 at the Space root."""
    return """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Startup Survival Simulator</title>
        <style>
          body {
            margin: 0;
            font-family: Georgia, 'Times New Roman', serif;
            background: linear-gradient(135deg, #f7f1e3 0%, #dff3e4 100%);
            color: #1f2a1f;
          }
          main {
            max-width: 760px;
            margin: 48px auto;
            padding: 32px;
            background: rgba(255, 255, 255, 0.88);
            border-radius: 20px;
            box-shadow: 0 18px 50px rgba(48, 76, 56, 0.15);
          }
          h1 {
            margin-top: 0;
            font-size: 2.5rem;
          }
          p, li {
            line-height: 1.6;
            font-size: 1rem;
          }
          .links a {
            display: inline-block;
            margin: 0 12px 12px 0;
            padding: 10px 16px;
            border-radius: 999px;
            text-decoration: none;
            background: #264653;
            color: #fff;
          }
          code {
            background: #edf6f9;
            padding: 2px 6px;
            border-radius: 6px;
          }
        </style>
      </head>
      <body>
        <main>
          <h1>Startup Survival Simulator</h1>
          <p>
            An OpenEnv-style startup management environment where an AI agent learns to balance
            growth, burn, churn, morale, and revenue through a standard <code>reset()</code> /
            <code>step()</code> / <code>state()</code> interface.
          </p>
          <p>
            This deployment is judge-friendly out of the box: no setup, no secrets, and all core
            evaluation endpoints are live.
          </p>
          <div class="links">
            <a href="/docs">Open API Docs</a>
            <a href="/tasks">View Tasks</a>
            <a href="/baseline">Run Baseline</a>
            <a href="/state">Current State</a>
          </div>
          <h2>Quick Start</h2>
          <ul>
            <li>Send <code>POST /reset</code> to start a fresh episode.</li>
            <li>Send <code>POST /step</code> with an action like <code>increase_marketing</code>.</li>
            <li>Use <code>GET /grader?task_name=survival</code> to score the current run.</li>
          </ul>
        </main>
      </body>
    </html>
    """


@app.post("/reset")
def reset_environment(request: ResetRequest | None = None) -> dict:
    """Reset the environment and return the initial state."""
    seed = request.seed if request else None
    return env.reset(seed=seed).model_dump()


@app.post("/step")
def step_environment(request: StepRequest) -> dict:
    """Execute one environment step."""
    try:
        return env.step(request.action.value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def get_state() -> dict:
    """Return the current state."""
    return env.state().model_dump()


@app.get("/tasks")
def list_tasks() -> dict:
    """Return task metadata and action schema."""
    return get_tasks()


@app.get("/grader")
def run_grader(task_name: str) -> dict:
    """Grade the current environment state for a task."""
    try:
        return grade(task_name, env.state().model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/baseline")
def execute_baseline(seed: int = 42) -> dict:
    """Run the deterministic baseline policy for all tasks."""
    return run_baseline(seed=seed)
