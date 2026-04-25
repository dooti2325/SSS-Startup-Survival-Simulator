---
title: Startup Survival Simulator
emoji: "🚀"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Startup Survival Simulator

An OpenEnv-style startup decision simulator where an agent learns how to survive and scale under constraints.

**Live Space:** https://huggingface.co/spaces/Loosebag/SSS-Startup-Survival-Simulator
**API Docs:** Publish your own Hugging Face Space first, then use `<your-space-url>/docs`

---

## Overview

Startup Survival Simulator is a real-world, OpenEnv-compliant environment exposing a standard `reset()` / `step()` / `state()` interface via FastAPI. An AI agent observes 10 startup metrics and chooses one of 7 actions each turn. The environment evolves through compounding effects on growth, revenue, product quality, morale, and cash — reflecting the real decisions an early-stage founder faces.

Episodes end when the startup **goes bankrupt**, **reaches 10,000 users**, or hits the **50-step timeout**.

---

## Environment Variables

Set these before running `inference.py`:

| Variable | Description | Example |
|---|---|---|
| `API_BASE_URL` | OpenAI-compatible LLM endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-7B-Instruct` |
| `HF_TOKEN` | Hugging Face API key | `hf_xxxxxxxxxxxx` |

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export HF_TOKEN="hf_xxxxxxxxxxxx"
```

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `cash` | `float` | Available cash in USD |
| `users` | `int` | Active users |
| `revenue` | `float` | Revenue this step in USD |
| `growth_rate` | `float [0,1]` | New-user multiplier |
| `burn_rate` | `float` | Operating cost per step in USD |
| `churn_rate` | `float [0,1]` | Fraction of users lost per step |
| `product_quality` | `float [0,1]` | Product quality score |
| `market_demand` | `float [0,1]` | External market demand score |
| `morale` | `float [0,1]` | Team morale score |
| `time_step` | `int` | Current step counter |

**Starting values:** cash=50,000 · users=100 · revenue=1,000 · growth_rate=0.08 · burn_rate=4,500 · churn_rate=0.03 · product_quality=0.55 · market_demand=0.60 · morale=0.70

---

## Action Space

| Action | Effect |
|---|---|
| `increase_marketing` | +growth_rate, +market_demand, ++burn_rate |
| `hire_engineer` | ++product_quality, +morale, +++burn_rate |
| `improve_product` | +product_quality, −churn_rate, +morale |
| `reduce_costs` | −burn_rate, −growth_rate, −morale |
| `pivot_market` | Random market_demand ± shift (high risk/reward) |
| `raise_funding` | Probabilistic +$30,000 cash (based on product quality & users) |
| `do_nothing` | −morale (tiny) |

---

## Tasks & Grading

| Task | Difficulty | Goal | Scoring Formula |
|---|---|---|---|
| `survival` | Easy | Survive 30 steps without bankruptcy | `time_step / 30 − cash_penalty` |
| `growth` | Medium | Reach 1,000 active users | `users / 1000 + sustainability_bonus` |
| `scaling` | Hard | Maximize revenue/burn efficiency | `efficiency × 0.7 + user_factor × 0.3` |

All scores are clamped to `[0.0, 1.0]`.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | HTML landing page — returns HTTP 200 |
| `POST` | `/reset` | Reset environment, optional `{"seed": 42}` body |
| `POST` | `/step` | Apply action, e.g. `{"action": "improve_product"}` |
| `GET` | `/state` | Current environment state |
| `GET` | `/tasks` | Task list + action schema |
| `GET` | `/grader?task_name=survival` | Score current state for a task |
| `GET` | `/baseline?seed=42` | Run deterministic baseline across all tasks |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Submission Files

The hackathon verifier expects these files in the repo root:

- `interface.py`
- `openenv.yaml`
- `requirements.txt`
- `Dockerfile`

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn api:app --host 0.0.0.0 --port 7860
```

Open:

- Local app: http://localhost:7860/
- Swagger docs: http://localhost:7860/docs

## Run Inference

Set these environment variables before running `inference.py`:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Example:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export HF_TOKEN="hf_xxxxxxxxxxxx"
python inference.py
```

## API Endpoints

- `GET /`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /grader?task_name=survival`
- `GET /baseline?seed=42`
- `GET /docs`

## Quick Test

```bash
pytest test_smoke.py -v
```

Optional validation:

```bash
bash validate_submission.sh
```

---

## Project Structure

```
├── api.py            # FastAPI app — all HTTP endpoints
├── env.py            # StartupEnv simulation logic
├── models.py         # Pydantic typed models (State, Action, StepResult, etc.)
├── grader.py         # Task graders — survival / growth / scaling
├── tasks.py          # Task metadata for /tasks endpoint
├── baseline.py       # Deterministic baseline policy
├── inference.py      # LLM inference script (hackathon evaluator entry point)
├── interface.py      # Repo-root compatibility interface for submission validators
├── test_smoke.py     # Pre-submission smoke tests
├── openenv.yaml      # OpenEnv spec manifest
├── Dockerfile        # Docker build for HF Spaces
└── requirements.txt  # Python dependencies
```
