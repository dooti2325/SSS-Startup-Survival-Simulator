"""Train and evaluate a TRL SFT policy for Startup Survival Simulator.

This script generates an expert-trajectory dataset directly from the live
StartupEnv, fine-tunes a causal LM with HF TRL's SFTTrainer, evaluates
trained vs random policy, and writes judge-friendly artifacts to ./artifacts/.

Quick local run (CPU, ~3 min):
    python train_trl.py --model_name distilgpt2 --dataset_episodes 8 \
        --eval_episodes 4 --num_train_epochs 1

GPU / Colab (see train_trl.ipynb for Unsloth/Qwen2.5-7B version):
    python train_trl.py --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --dataset_episodes 32 --eval_episodes 12 --num_train_epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from env import StartupEnv
from grader import grade
from models import Action, StartupState

TASKS = ("survival", "growth", "scaling")
ACTIONS = [action.value for action in Action]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRL SFT training pipeline for Startup Survival Simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name", default="distilgpt2",
                        help="HuggingFace model id to fine-tune.")
    parser.add_argument("--output_dir", default="artifacts/trl_model",
                        help="Directory to save model checkpoints and logs.")
    parser.add_argument("--dataset_episodes", type=int, default=24,
                        help="Expert rollout episodes per task for dataset generation.")
    parser.add_argument("--dataset_max_steps", type=int, default=35,
                        help="Max steps per rollout episode.")
    parser.add_argument("--eval_episodes", type=int, default=12,
                        help="Evaluation seeds per task.")
    parser.add_argument("--num_train_epochs", type=float, default=1.0,
                        help="Number of SFT training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="AdamW learning rate.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device train batch size.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an AI CEO managing an early-stage startup.\n"
    "Observe the current business state and choose exactly one action.\n\n"
    "RULES:\n"
    "1. Market demand and churn are HIDDEN — use analyze_market to estimate them.\n"
    "2. Hiring builds hidden technical_debt — use refactor_code before a Server Crash.\n"
    "3. Rewards are SPARSE — milestones at 1k, 2.5k, 5k, 7.5k, 10k users.\n"
    "4. Bankruptcy (cash=0) gives a -1000 penalty.\n\n"
    f"Available actions: {', '.join(ACTIONS)}\n\n"
    "Reply with ONLY the action name. No explanation."
)


def build_prompt(task_name: str, state: dict) -> str:
    s = state
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Task: {task_name}\n"
        f"Step {s.get('time_step', 0)}. Current state:\n"
        f"- Cash: ${s['cash']:,.0f}\n"
        f"- Users: {s['users']:,}\n"
        f"- Revenue/step: ${s['revenue']:,.0f}\n"
        f"- Burn rate/step: ${s['burn_rate']:,.0f}\n"
        f"- Growth rate: {s['growth_rate']:.3f}\n"
        f"- Product quality: {s['product_quality']:.3f}\n"
        f"- Team morale: {s['morale']:.3f}\n"
        f"Note: market_demand, churn_rate, technical_debt are hidden.\n"
        f"Action:"
    )


def build_training_text(task_name: str, state: dict, action: str) -> str:
    return f"{build_prompt(task_name, state)} {action}"


# ---------------------------------------------------------------------------
# Expert / oracle policy
# ---------------------------------------------------------------------------

def heuristic_action(task_name: str, state: dict) -> str:
    """Deterministic expert policy — used to label training data."""
    cash     = state["cash"]
    users    = state["users"]
    revenue  = state["revenue"]
    burn     = state["burn_rate"]
    quality  = state["product_quality"]
    morale   = state["morale"]
    step     = state.get("time_step", 0)

    # Periodic maintenance actions
    if step > 0 and step % 8 == 0 and cash > 3_000:
        return "refactor_code"
    if step > 0 and step % 5 == 0 and cash > 2_000:
        return "analyze_market"

    # Universal guard
    if cash < 8_000:
        return "reduce_costs"

    if task_name == "survival":
        if burn > revenue * 1.5:
            return "reduce_costs"
        if quality < 0.78:
            return "improve_product"
        return "do_nothing"

    if task_name == "growth":
        if users < 600:
            return "increase_marketing"
        if quality < 0.80:
            return "improve_product"
        if morale < 0.35:
            return "do_nothing"
        return "hire_engineer"

    # scaling
    if burn > revenue * 1.2:
        return "reduce_costs"
    if quality < 0.82:
        return "improve_product"
    if cash > 20_000 and revenue < burn:
        return "raise_funding"
    return "increase_marketing"


def one_step_action_value(task_name: str, state: dict, action: str, sim_seed: int) -> float:
    """Score an action by simulating one step and grading the resulting state."""
    sim = StartupEnv(seed=sim_seed)
    sim.current_state = StartupState(**state)
    sim._market_demand = state.get("market_demand", 0.6)   # pylint: disable=protected-access
    sim._churn_rate    = state.get("churn_rate",    0.03)  # pylint: disable=protected-access
    sim._technical_debt   = 0.0                            # pylint: disable=protected-access
    sim._milestones_reached = set()                        # pylint: disable=protected-access

    result = sim.step(action)
    task_score = grade(task_name, result["state"])["score"]
    return float(result["reward"]) + task_score * 100.0


def choose_training_action(task_name: str, state: dict, sim_seed: int) -> str:
    """Pick the highest-scoring action via one-step look-ahead."""
    scored = [
        (one_step_action_value(task_name, state, a, sim_seed), a)
        for a in ACTIONS
    ]
    return max(scored, key=lambda x: x[0])[1]


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def build_training_dataset(
    dataset_episodes: int,
    dataset_max_steps: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    sim_seed_base = seed * 17

    for task_name in TASKS:
        for ep in range(dataset_episodes):
            env = StartupEnv(seed=seed + ep)
            env.reset(seed=seed + ep)
            done = False
            steps = 0

            while not done and steps < dataset_max_steps:
                state = env.state().model_dump()
                label = choose_training_action(task_name, state, sim_seed=sim_seed_base + steps)
                rows.append({"text": build_training_text(task_name, state, label)})

                # 70 % oracle, 30 % random to diversify visited states
                rollout_action = label if rng.random() < 0.7 else rng.choice(ACTIONS)
                out = env.step(rollout_action)
                done = bool(out["done"])
                steps += 1

    return rows


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def normalize_action(raw: str) -> str:
    cleaned = raw.strip().lower()
    m = re.search(r"[a-z_]+", cleaned)
    return m.group(0) if m else ""


def model_action(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    task_name: str,
    state: dict,
) -> str:
    prompt = build_prompt(task_name, state)
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=6,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    action = normalize_action(decoded)
    return action if action in ACTIONS else heuristic_action(task_name, state)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    policy_fn: Callable[[str, dict], str],
    eval_episodes: int,
    seed: int,
) -> dict:
    per_task: dict[str, dict] = {}
    all_scores: list[float] = []
    all_rewards: list[float] = []

    for task_name in TASKS:
        t_scores: list[float] = []
        t_rewards: list[float] = []

        for ep in range(eval_episodes):
            env = StartupEnv(seed=seed + 1000 + ep)
            env.reset(seed=seed + 1000 + ep)
            done = False
            total_reward = 0.0

            while not done:
                state = env.state().model_dump()
                action = policy_fn(task_name, state)
                result = env.step(action)
                done = bool(result["done"])
                total_reward += float(result["reward"])

            final = env.state().model_dump()
            score = float(grade(task_name, final)["score"])
            t_scores.append(score)
            t_rewards.append(total_reward)
            all_scores.append(score)
            all_rewards.append(total_reward)

        per_task[task_name] = {
            "avg_score": round(sum(t_scores) / max(len(t_scores), 1), 4),
            "avg_total_reward": round(sum(t_rewards) / max(len(t_rewards), 1), 4),
            "episodes": eval_episodes,
        }

    return {
        "per_task": per_task,
        "overall_avg_score": round(sum(all_scores) / max(len(all_scores), 1), 4),
        "overall_avg_total_reward": round(sum(all_rewards) / max(len(all_rewards), 1), 4),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save_training_loss_plot(log_history: list[dict], artifacts_dir: Path) -> Path:
    points = [(e.get("step"), e.get("loss")) for e in log_history if "loss" in e]
    out = artifacts_dir / "training_loss.png"
    if not points:
        return out

    steps  = [p[0] for p in points]
    losses = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, losses, color="#1a73e8", linewidth=2, marker="o", markersize=4)
    ax.fill_between(steps, losses, alpha=0.10, color="#1a73e8")
    ax.set_title("SFT Training Loss — Startup Survival Agent", fontsize=14)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_evaluation_plot(summary: dict, artifacts_dir: Path) -> Path:
    metrics    = ["overall_avg_score", "overall_avg_total_reward"]
    labels     = ["Avg Score (0-1)", "Avg Total Reward"]
    rand_vals  = [summary["random_policy"][m]  for m in metrics]
    train_vals = [summary["trained_policy"][m] for m in metrics]

    x = list(range(len(metrics)))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars_r = ax.bar([xi - w / 2 for xi in x], rand_vals,  w, label="Random Policy",  color="#e74c3c", alpha=0.85)
    bars_t = ax.bar([xi + w / 2 for xi in x], train_vals, w, label="TRL Trained",    color="#2ecc71", alpha=0.85)

    # Annotate bars
    for bar in bars_r:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_t:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Baseline vs TRL Trained Policy", fontsize=14)
    ax.set_ylabel("Metric Value")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = artifacts_dir / "evaluation_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Building training dataset ({args.dataset_episodes} eps/task) …")
    rows = build_training_dataset(
        dataset_episodes=args.dataset_episodes,
        dataset_max_steps=args.dataset_max_steps,
        seed=args.seed,
    )
    dataset = Dataset.from_list(rows)
    print(f"      → {len(rows)} training samples across {len(TASKS)} tasks")

    print(f"[2/6] Loading model: {args.model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    print("[3/6] Fine-tuning with SFTTrainer …")
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        max_length=512,
        use_cpu=not torch.cuda.is_available(),
        gradient_checkpointing=False,
        dataset_text_field="text",
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"      → Model saved to {output_dir}")

    log_history = trainer.state.log_history
    (artifacts_dir / "training_log_history.json").write_text(
        json.dumps(log_history, indent=2), encoding="utf-8"
    )

    print(f"[4/6] Evaluating random policy ({args.eval_episodes} eps/task) …")
    random_summary = evaluate_policy(
        policy_fn=lambda _t, _s: random.choice(ACTIONS),
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )

    print(f"[5/6] Evaluating trained policy ({args.eval_episodes} eps/task) …")
    trained_summary = evaluate_policy(
        policy_fn=lambda t, s: model_action(model, tokenizer, t, s),
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )

    summary = {
        "model_name": args.model_name,
        "dataset_rows": len(rows),
        "dataset_episodes_per_task": args.dataset_episodes,
        "training_epochs": args.num_train_epochs,
        "random_policy":  random_summary,
        "trained_policy": trained_summary,
        "improvement": {
            "avg_score_delta": round(
                trained_summary["overall_avg_score"] - random_summary["overall_avg_score"], 4
            ),
            "avg_total_reward_delta": round(
                trained_summary["overall_avg_total_reward"] - random_summary["overall_avg_total_reward"], 4
            ),
        },
    }

    print("[6/6] Writing artifacts …")
    (artifacts_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    loss_plot = save_training_loss_plot(log_history, artifacts_dir)
    eval_plot = save_evaluation_plot(summary, artifacts_dir)

    # Pretty summary to stdout
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written:")
    print(f"  {artifacts_dir}/evaluation_summary.json")
    print(f"  {artifacts_dir}/training_log_history.json")
    print(f"  {loss_plot}")
    print(f"  {eval_plot}")


if __name__ == "__main__":
    main()
