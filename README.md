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

## What This Project Does

The environment exposes standard API actions for:

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

Run stress/debug checks:

```bash
python sss_stress_debug.py
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
