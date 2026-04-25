"""Visualize baseline vs trained policy performance from demo_outputs data.

Usage:
    python sss_visualize_demo.py
"""

from __future__ import annotations

import json
import argparse
from collections import Counter
from pathlib import Path

DEFAULT_INPUT = Path("demo_outputs") / "demo_results.json"
DEFAULT_OUTPUT = Path("demo_outputs") / "policy_comparison_plots.png"


def _load_demo_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _extract_action_counts(replay_block: dict) -> Counter:
    trajectory = replay_block.get("trajectory", [])
    actions = [step.get("info", {}).get("action", "unknown") for step in trajectory]
    return Counter(actions)


def build_plots(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> Path:
    import matplotlib.pyplot as plt

    data = _load_demo_results(input_path)

    baseline_metrics = data["baseline_metrics"]
    trained_metrics = data["trained_metrics"]
    baseline_rewards = baseline_metrics["episode_rewards"]
    trained_rewards = trained_metrics["episode_rewards"]

    baseline_survival = baseline_metrics["survival_rate"]
    trained_survival = trained_metrics["survival_rate"]

    baseline_actions = _extract_action_counts(data["same_seed_replay"]["baseline"])
    trained_actions = _extract_action_counts(data["same_seed_replay"]["trained"])
    all_actions = sorted(set(baseline_actions.keys()) | set(trained_actions.keys()))
    baseline_action_vals = [baseline_actions.get(action, 0) for action in all_actions]
    trained_action_vals = [trained_actions.get(action, 0) for action in all_actions]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SSS Policy Comparison (Baseline vs Trained)", fontsize=14, fontweight="bold")

    # 1) Reward distribution
    axes[0].hist(baseline_rewards, bins=10, alpha=0.65, label="Baseline", color="#d62728")
    axes[0].hist(trained_rewards, bins=10, alpha=0.65, label="Trained", color="#2ca02c")
    axes[0].set_title("Reward Distribution")
    axes[0].set_xlabel("Episode Total Reward")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # 2) Survival rate comparison
    axes[1].bar(
        ["Baseline", "Trained"],
        [baseline_survival, trained_survival],
        color=["#d62728", "#2ca02c"],
        alpha=0.8,
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Survival Rate Comparison")
    axes[1].set_ylabel("Survival Rate")
    for idx, val in enumerate([baseline_survival, trained_survival]):
        axes[1].text(idx, val + 0.02, f"{val:.2f}", ha="center", va="bottom")

    # 3) Decision frequency differences
    x = range(len(all_actions))
    width = 0.4
    axes[2].bar([i - width / 2 for i in x], baseline_action_vals, width=width, label="Baseline", color="#d62728")
    axes[2].bar([i + width / 2 for i in x], trained_action_vals, width=width, label="Trained", color="#2ca02c")
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(all_actions, rotation=35, ha="right")
    axes[2].set_title("Decision Frequency (Replay Seed)")
    axes[2].set_ylabel("Action Count")
    axes[2].legend()

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def validate_input(input_path: Path = DEFAULT_INPUT) -> None:
    data = _load_demo_results(input_path)
    required_top = {"baseline_metrics", "trained_metrics", "same_seed_replay"}
    missing = required_top - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys in demo data: {sorted(missing)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize baseline vs trained policy from demo_outputs JSON.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to demo_results.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to output PNG")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate demo_outputs data schema without plotting.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    validate_input(input_path)
    if args.validate_only:
        print(f"Validated input data: {input_path}")
    else:
        saved = build_plots(input_path=input_path, output_path=output_path)
        print(f"Saved visualization: {saved}")
