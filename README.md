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

## Submission Links

- Hugging Face Space URL: https://huggingface.co/spaces/Loosebag/SSS-Startup-Survival-Simulator
- Colab Notebook: https://colab.research.google.com/github/DivyankLosse/SSS-Startup-Survival-Simulator/blob/main/train_trl.ipynb
- Code Repository: https://github.com/DivyankLosse/SSS-Startup-Survival-Simulator

<<<<<<< HEAD
Startup Survival Simulator is a real-world, OpenEnv-compliant environment exposing a standard `reset()` / `step()` / `state()` interface via FastAPI. An AI agent observes 8 startup metrics and chooses one of 9 actions each turn. 

The environment is designed to test:
- **Partial Observability:** Critical metrics like `market_demand` and `churn_rate` are hidden. The agent must orchestrate multi-step workflows by using the `analyze_market` tool (API) to pierce this fog.
- **Mistakes & Recovery:** Rapid growth builds hidden `technical_debt`. Without using the `refactor_code` action, the startup will experience a massive "Server Crash".
- **Sparse Rewards:** The agent receives 0 reward per step, requiring successful scaling to hit massive milestone payouts.
=======
## What This Project Does

The environment exposes standard API actions for:
>>>>>>> c2a96c6707af8220d912d11c1f4810e9d70e46cc

- resetting the simulator
- stepping the environment with one startup action
- reading current state
- listing tasks
- grading performance
- running a simple baseline

Episodes end when the startup:

- goes bankrupt
- reaches 10,000 users
- hits the 50-step limit

## Main Actions

- `increase_marketing`
- `hire_engineer`
- `improve_product`
- `reduce_costs`
- `pivot_market`
- `raise_funding`
- `do_nothing`

## Required Submission Files

- `api.py`
- `env.py`
- `models.py`
- `grader.py`
- `tasks.py`
- `baseline.py`
- `inference.py`
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
<<<<<<< HEAD
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
| `product_quality` | `float [0,1]` | Product quality score |
| `morale` | `float [0,1]` | Team morale score |
| `time_step` | `int` | Current step counter |

*Note: `market_demand`, `churn_rate`, and `technical_debt` are explicitly hidden from the state to enforce World Modeling.*

**Starting values:** cash=50,000 · users=100 · revenue=1,000 · growth_rate=0.08 · burn_rate=4,500 · product_quality=0.55 · morale=0.70

---

## Action Space

| Action | Effect |
|---|---|
| `increase_marketing` | +growth_rate, +market_demand, ++burn_rate |
| `hire_engineer` | ++product_quality, +morale, +++burn_rate, +tech_debt |
| `improve_product` | +product_quality, −churn_rate, +morale |
| `reduce_costs` | −burn_rate, −growth_rate, −morale |
| `pivot_market` | Random market_demand ± shift (high risk/reward) |
| `raise_funding` | Probabilistic +$30,000 cash (based on product quality & users) |
| `analyze_market` | Tool action: Cost $1,000. Returns noisy info about hidden market_demand and churn_rate |
| `refactor_code` | Recovery action: Cost $2,500. Reduces hidden tech_debt to prevent Server Crashes |
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
- `inference.py`
- `openenv.yaml`
- `requirements.txt`

## Running the Inference Script

```bash
pip install -r requirements.txt
=======
>>>>>>> c2a96c6707af8220d912d11c1f4810e9d70e46cc
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

## Hackathon Pipeline (Baseline vs Trained)

This repository includes a reproducible hackathon flow:

- Environment: `sss_hackathon_env.py`
- Reward + Verifier: `sss_reward_verifier.py`
- Training loop: `sss_training.py`
- Demo runner: `sss_demo.py`
- Stress/debug checks: `sss_stress_debug.py`
- Scenario support: `standard`, `recession`, `competition`
- Architecture doc: `HACKATHON_ARCHITECTURE.md`

Run the full hackathon demo:

```bash
python sss_demo.py
```
<<<<<<< HEAD
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
├── train_trl.ipynb   # Unsloth TRL fine-tuning notebook
├── Dockerfile        # Docker build for HF Spaces
└── requirements.txt  # Python dependencies
=======

Run stress/debug checks:

```bash
python sss_stress_debug.py
>>>>>>> c2a96c6707af8220d912d11c1f4810e9d70e46cc
```

Run visualization from demo outputs:

```bash
python sss_visualize_demo.py
```

Generated artifacts:

- `demo_outputs/demo_results.json`
- `demo_outputs/trained_policy_qtable.json`

Scenario metrics are included in `demo_results.json` under:

- `scenario_results.recession`
- `scenario_results.competition`

## Notes For Submission

- Keep all submission links updated in this README.
- Push `train_trl.ipynb` so the Colab link stays valid.
