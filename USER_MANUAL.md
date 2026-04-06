# Startup Survival Simulator — User Manual

> **Version 1.1.0** · Last updated: April 2026  
> Live API: https://dootisaha25-startup-survival-simulator.hf.space

---

## Table of Contents

1. [What Is This?](#1-what-is-this)
2. [Quick Start — 60 Seconds](#2-quick-start--60-seconds)
3. [Environment Variables](#3-environment-variables)
4. [Running the Inference Script](#4-running-the-inference-script)
5. [Using the API](#5-using-the-api)
6. [Understanding the Simulation](#6-understanding-the-simulation)
7. [Tasks and Scoring](#7-tasks-and-scoring)
8. [Writing Your Own Agent](#8-writing-your-own-agent)
9. [Local Development](#9-local-development)
10. [Docker Deployment](#10-docker-deployment)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What Is This?

The **Startup Survival Simulator** is an AI evaluation environment where a language model agent plays the role of a startup CEO. Each "turn" the agent observes 10 business metrics and picks one of 7 strategic actions. The environment evolves through compounding effects — good decisions compound into growth, bad ones spiral into bankruptcy.

It follows the **OpenEnv** standard: a clean `reset() → step() → state()` loop exposed over HTTP, making it trivial for evaluators and agents to interact with it programmatically.

```
┌──────────────┐     POST /reset     ┌──────────────────┐
│  Your Agent  │ ──────────────────► │  StartupEnv      │
│  (LLM/rule)  │ ◄────────────────── │  (FastAPI)       │
│              │   initial state     │                  │
│              │     POST /step      │  Tracks: cash,   │
│              │ ──────────────────► │  users, revenue, │
│              │ ◄────────────────── │  morale, burn,   │
│              │  {state,reward,done}│  churn, quality  │
└──────────────┘                     └──────────────────┘
```

---

## 2. Quick Start — 60 Seconds

### Against the live HF Space (no install needed)

```bash
# 1. Reset the environment
curl -X POST "https://dootisaha25-startup-survival-simulator.hf.space/reset" \
     -H "Content-Type: application/json" -d '{}'

# 2. Take one action
curl -X POST "https://dootisaha25-startup-survival-simulator.hf.space/step" \
     -H "Content-Type: application/json" -d '{"action": "improve_product"}'

# 3. Check score
curl "https://dootisaha25-startup-survival-simulator.hf.space/grader?task_name=survival"
```

### Locally

```bash
git clone https://github.com/dooti2325/SSS-Startup-Survival-Simulator.git
cd SSS-Startup-Survival-Simulator
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 7860
# Open http://localhost:7860/docs
```

---

## 3. Environment Variables

All three variables are **required** to run `Inference.py`. The API server itself runs without them (no LLM calls at server startup).

| Variable | Purpose | Where to get it |
|---|---|---|
| `API_BASE_URL` | Base URL of an OpenAI-compatible LLM endpoint | Your provider's docs |
| `MODEL_NAME` | Model identifier string | Your provider's model list |
| `HF_TOKEN` | Hugging Face API key (used as the `api_key` for the OpenAI client) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

### Setting variables

**Windows (PowerShell):**
```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME   = "Qwen/Qwen2.5-7B-Instruct"
$env:HF_TOKEN     = "hf_xxxxxxxxxxxxxxxxxxxx"
```

**Linux / macOS:**
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

### Recommended free models (HF Inference API)

| Model | Speed | Quality | Notes |
|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | Fast | Good | Default recommendation |
| `HuggingFaceH4/zephyr-7b-beta` | Fast | Good | Strong instruction following |
| `meta-llama/Meta-Llama-3-8B-Instruct` | Medium | Great | Best reasoning, may require access |

> **HF Router base URL:** `https://router.huggingface.co/v1`

---

## 4. Running the Inference Script

`Inference.py` is the **official evaluator entry point**. It runs 3 tasks end-to-end and prints structured logs.

```bash
python Inference.py
```

### What it does

1. Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from environment
2. Creates an `OpenAI` client pointed at your LLM endpoint
3. For each task (`survival`, `growth`, `scaling`):
   - Resets the environment with seed 42
   - Loops: sends current state to LLM → gets action → calls `env.step()`
   - Stops when `done=True` (bankrupt / 10k users / 50 steps)
   - Prints final score via grader

### Output format (mandatory)

The evaluator parses stdout **token-by-token** — every field name, `=` separator, spacing, and lowercase boolean must be exact.

```
[START] task=survival env=startup-survival-simulator model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action=improve_product reward=12.50 done=false error=null
[STEP] step=2 action=raise_funding reward=8.30 done=false error=null
...
[END] success=true steps=30 rewards=12.50,8.30,...
[START] task=growth env=startup-survival-simulator model=Qwen/Qwen2.5-7B-Instruct
...
[END] success=false steps=50 rewards=...
[START] task=scaling env=startup-survival-simulator model=Qwen/Qwen2.5-7B-Instruct
...
[END] success=true steps=22 rewards=...
```

| Field | Format | Notes |
|---|---|---|
| `task` | string | Task name: `survival`, `growth`, `scaling` |
| `env` | string | Always `startup-survival-simulator` |
| `model` | string | Value of `MODEL_NAME` env var |
| `step` | integer | 1-indexed |
| `action` | string | Exact action name, e.g. `improve_product` |
| `reward` | `0.00` | 2 decimal places |
| `done` | `true`/`false` | **Lowercase** boolean |
| `error` | string or `null` | Raw error message or `null` |
| `success` | `true`/`false` | **Lowercase** — `true` if score ≥ 0.5 |
| `rewards` | `r1,r2,...` | Comma-separated, 2 decimal places each |

### Typical runtime

- ~1–3 minutes for all 3 tasks with a fast model
- Well under the 20-minute limit on 2 vCPU / 8 GB machines

---

## 5. Using the API

The full interactive docs are at `/docs`. Below is a complete reference.

### `GET /`
Health check. Returns an HTML landing page with HTTP 200. Used by the hackathon validator to confirm the Space is running.

---

### `POST /reset`
Resets the environment to the initial state.

**Request body** (optional):
```json
{ "seed": 42 }
```
Omitting the body or sending `{}` uses the default seed (42).

**Response:**
```json
{
  "cash": 50000.0,
  "users": 100,
  "revenue": 1000.0,
  "growth_rate": 0.08,
  "burn_rate": 4500.0,
  "churn_rate": 0.03,
  "product_quality": 0.55,
  "market_demand": 0.6,
  "morale": 0.7,
  "time_step": 0
}
```

---

### `POST /step`
Applies one action and advances the simulation by one time step.

**Request body** (required):
```json
{ "action": "improve_product" }
```

**Valid action values:**
```
increase_marketing | hire_engineer | improve_product |
reduce_costs | pivot_market | raise_funding | do_nothing
```

**Response:**
```json
{
  "state": { "cash": 46800.0, "users": 138, ... },
  "reward": 12.5,
  "done": false,
  "info": {
    "action": "improve_product",
    "acquired_users": 42,
    "lost_users": 3,
    "net_users": 39,
    "revenue_delta": 560.0
  }
}
```

`done: true` means the episode is over. Check `info.reason` for `"bankrupt"`, `"success"`, or `"timeout"`.

---

### `GET /state`
Returns the current state without stepping.

---

### `GET /tasks`
Returns task metadata and the action schema.

```json
{
  "tasks": [
    { "id": "survival", "difficulty": "easy", "name": "Survive 30 Steps", ... },
    { "id": "growth",   "difficulty": "medium", ... },
    { "id": "scaling",  "difficulty": "hard", ... }
  ],
  "action_schema": {
    "type": "object",
    "required": ["action"],
    "properties": { "action": { "type": "string", "enum": [...] } }
  }
}
```

---

### `GET /grader?task_name=<task>`
Scores the **current** environment state for the specified task. Call after an episode ends.

**Query param:** `task_name` = `survival` | `growth` | `scaling`

**Response:**
```json
{ "score": 0.87 }
```

---

### `GET /baseline?seed=42`
Runs the built-in deterministic baseline agent across all three tasks and returns scores.

**Response:**
```json
{
  "survival": { "score": 0.73 },
  "growth":   { "score": 0.31 },
  "scaling":  { "score": 0.58 }
}
```

---

## 6. Understanding the Simulation

### Starting State

Every episode begins with a seed-42 startup: $50k cash, 100 users, $1k revenue, 8% growth rate, $4.5k burn/step.

### Each Step

1. Action effects applied (burn changes, stat boosts, etc.)
2. Market noise added (±2% demand, ±0.3% churn)
3. New users acquired = `users × growth_rate × quality × demand × morale`
4. Churned users subtracted
5. Revenue = `users × ARPU × market_demand` (ARPU = $14–$20 based on quality)
6. `cash = cash + revenue − burn_rate`
7. Reward computed, done-condition checked

### Action Effects Cheatsheet

| Action | burn_rate | growth_rate | product_quality | churn_rate | morale | cash |
|---|---|---|---|---|---|---|
| `increase_marketing` | +1,200 | +0.03 | — | — | — | — |
| `hire_engineer` | +2,200 | — | +0.07 | — | +0.03 | — |
| `improve_product` | — | — | +0.05 | −0.006 | +0.02 | — |
| `reduce_costs` | −1,600 | −0.015 | — | — | −0.06 | — |
| `pivot_market` | — | — | ±random | +0.01 churn | — | — |
| `raise_funding` | +600 | — | — | — | ±0.03 | +30k (probabilistic) |
| `do_nothing` | — | — | — | — | −0.01 | — |

### Episode End Conditions

| Condition | `done` | `info.reason` |
|---|---|---|
| `cash <= 0` | `true` | `"bankrupt"` |
| `users >= 10,000` | `true` | `"success"` |
| `time_step >= 50` | `true` | `"timeout"` |

---

## 7. Tasks and Scoring

### Survival (Easy)
**Goal:** Keep the startup alive for at least 30 steps.

```
score = clamp(time_step / 30.0 − cash_penalty)
cash_penalty = 0.2 if cash < $5,000 else 0.0
```

**Strategy tips:**
- Prioritize `reduce_costs` early if cash is draining
- `improve_product` is cheap and reduces churn — great default action
- Avoid `hire_engineer` in the first 10 steps (burn is too high)

---

### Growth (Medium)
**Goal:** Reach 1,000 active users.

```
score = clamp(users / 1000.0 + sustainability_bonus)
sustainability_bonus = 0.1 if burn_rate <= revenue × 1.2 else 0.0
```

**Strategy tips:**
- Alternate `increase_marketing` and `improve_product` to grow users while reducing churn
- Watch burn — `hire_engineer` + `increase_marketing` together will drain cash fast
- `raise_funding` when `product_quality > 0.6` has a high success probability

---

### Scaling (Hard)
**Goal:** Maximize revenue/burn efficiency at the end of the episode.

```
efficiency = revenue / max(burn_rate, 1.0)
user_factor = users / 2000.0
score = clamp(efficiency × 0.7 + user_factor × 0.3)
```

**Strategy tips:**
- Revenue scales with users × product_quality × market_demand — optimize all three
- Keep `burn_rate` in check: `hire_engineer` is only worth it if product_quality pays off in revenue
- Score is measured at the **final** state — a strong finish matters more than early steps

---

## 8. Writing Your Own Agent

You can replace the LLM calls in `Inference.py` with any agent — rule-based, ML model, or another LLM.

### Minimal agent loop

```python
import os
from openai import OpenAI
from env import StartupEnv
from grader import grade

env = StartupEnv(seed=42)
env.reset(seed=42)

client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["HF_TOKEN"]
)

done = False
step = 1
while not done:
    state = env.state().model_dump()
    
    # --- Replace this block with your own logic ---
    response = client.chat.completions.create(
        model=os.environ["MODEL_NAME"],
        messages=[
            {"role": "system", "content": "Pick one action: increase_marketing, hire_engineer, improve_product, reduce_costs, pivot_market, raise_funding, do_nothing. Reply with just the action name."},
            {"role": "user", "content": str(state)}
        ],
        temperature=0.0,
        max_tokens=10
    )
    action = response.choices[0].message.content.strip()
    # -----------------------------------------------
    
    result = env.step(action)
    done = result["done"]
    step += 1

score = grade("survival", env.state().model_dump())["score"]
print(f"Final score: {score}")
```

### Rule-based agent example

```python
def rule_agent(state: dict) -> str:
    if state["cash"] < 10_000:
        return "reduce_costs"
    if state["churn_rate"] > 0.08:
        return "improve_product"
    if state["users"] < 500:
        return "increase_marketing"
    if state["product_quality"] > 0.7 and state["users"] < 1000:
        return "raise_funding"
    return "improve_product"
```

### Using the API from Python (against live Space)

```python
import requests

BASE = "https://dootisaha25-startup-survival-simulator.hf.space"

# Reset
state = requests.post(f"{BASE}/reset", json={"seed": 42}).json()

# Step loop
done = False
while not done:
    action = rule_agent(state)       # your agent logic
    result = requests.post(f"{BASE}/step", json={"action": action}).json()
    state = result["state"]
    done = result["done"]

# Score
score = requests.get(f"{BASE}/grader", params={"task_name": "survival"}).json()
print(score)  # {"score": 0.87}
```

---

## 9. Local Development

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 7860 --reload
```

`--reload` enables hot-reload for development.

### Run smoke tests

```bash
pytest test_smoke.py -v
```

Expected output:
```
test_smoke.py::test_root_returns_ok                  PASSED
test_smoke.py::test_reset_step_state_flow            PASSED
test_smoke.py::test_tasks_endpoint_exposes_schema    PASSED
test_smoke.py::test_grader_and_baseline_are_valid    PASSED
4 passed in 0.74s
```

### Run inference locally

```powershell
# Windows PowerShell
$env:API_BASE_URL = "https://api-inference.huggingface.co/v1"
$env:MODEL_NAME   = "mistralai/Mistral-7B-Instruct-v0.3"
$env:HF_TOKEN     = "hf_xxxx"
python Inference.py
```

---

## 10. Docker Deployment

### Build and run locally

```bash
docker build -t startup-survival-simulator .

docker run -p 7860:7860 \
  -e API_BASE_URL="https://api-inference.huggingface.co/v1" \
  -e MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" \
  -e HF_TOKEN="hf_xxxx" \
  startup-survival-simulator
```

Then visit: http://localhost:7860

### Deploy to Hugging Face Spaces

```bash
# Add HF remote (one time)
git remote add space https://<username>:<HF_TOKEN>@huggingface.co/spaces/<username>/startup-survival-simulator

# Push
git push space master:main --force
```

The Space rebuilds automatically. Build takes ~2 minutes.

---

## 11. Troubleshooting

### `Inference.py` exits immediately with no output

- Check that `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are set
- Run `echo $env:HF_TOKEN` (PowerShell) or `echo $HF_TOKEN` (bash)

### LLM returns an invalid action

`Inference.py` has a fallback: any response not in the valid action list is automatically replaced with `do_nothing`. The run won't crash.

### `POST /step` returns 400

The action value is invalid. Use one of the exact strings:
```
increase_marketing, hire_engineer, improve_product,
reduce_costs, pivot_market, raise_funding, do_nothing
```

### HF Space shows "Building" for a long time

Normal — Docker builds take 1–3 minutes. Watch the build logs in the Space settings on HF.

### `raise_funding` keeps returning `funding_raised: 0`

Funding success is probabilistic, based on `product_quality` and `users`:
```
probability = 0.2 + (product_quality × 0.3) + min(0.35, users / 5000)
```
Improve your product first (`improve_product` × 3–4 steps) before trying to raise.

### `pytest test_smoke.py` fails with import error

```bash
pip install pytest httpx fastapi
```

### Score is always 0.0 for `scaling`

Check that `burn_rate` isn't runaway. If `revenue < burn_rate`, efficiency is < 1.0. Use `reduce_costs` and grow users before scoring.

---

*For more information, see the [Swagger docs](https://dootisaha25-startup-survival-simulator.hf.space/docs) or open an issue on [GitHub](https://github.com/dooti2325/SSS-Startup-Survival-Simulator).*
