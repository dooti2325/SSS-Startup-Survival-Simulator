"""Reproducible baseline-vs-trained demo script for judges."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from sss_hackathon_env import StartupSurvivalEnv
from sss_reward_verifier import verify_episode
from sss_training import (
    build_greedy_policy,
    build_random_policy,
    evaluate_policy_in_scenario,
    run_episode,
    save_artifacts,
    train_q_learning,
)


def _replay(seed: int, policy_fn) -> Dict[str, object]:
    env = StartupSurvivalEnv(seed=seed)
    total_reward, trajectory = run_episode(env, policy_fn, seed=seed)
    verdict = verify_episode(trajectory)
    return {
        "seed": seed,
        "total_reward": total_reward,
        "steps": len(trajectory),
        "verdict": verdict,
        "trajectory": trajectory,
    }


def run_demo() -> Dict[str, object]:
    scenario_seeds: List[int] = list(range(1, 41))
    replay_seed = 19

    baseline_metrics = evaluate_policy_in_scenario(
        build_random_policy(seed=2026),
        scenario_seeds,
        scenario="standard",
    )

    artifacts = train_q_learning(episodes=350, seed=2026)
    trained_policy = build_greedy_policy(artifacts)
    trained_metrics = evaluate_policy_in_scenario(trained_policy, scenario_seeds, scenario="standard")

    baseline_replay = _replay(replay_seed, build_random_policy(seed=2026))
    trained_replay = _replay(replay_seed, trained_policy)

    reward_lift = round(trained_metrics["avg_total_reward"] - baseline_metrics["avg_total_reward"], 4)
    survival_lift = round(trained_metrics["survival_rate"] - baseline_metrics["survival_rate"], 4)
    verifier_lift = round(trained_metrics["verifier_pass_rate"] - baseline_metrics["verifier_pass_rate"], 4)

    results = {
        "baseline_metrics": baseline_metrics,
        "trained_metrics": trained_metrics,
        "improvement": {
            "avg_reward_lift": reward_lift,
            "survival_rate_lift": survival_lift,
            "verifier_pass_rate_lift": verifier_lift,
        },
        "same_seed_replay": {
            "seed": replay_seed,
            "baseline": baseline_replay,
            "trained": trained_replay,
        },
    }

    scenario_results = {}
    for scenario_name in ("recession", "competition"):
        baseline_scenario = evaluate_policy_in_scenario(
            build_random_policy(seed=2026),
            scenario_seeds,
            scenario=scenario_name,
        )
        trained_scenario = evaluate_policy_in_scenario(trained_policy, scenario_seeds, scenario=scenario_name)
        scenario_results[scenario_name] = {
            "baseline": baseline_scenario,
            "trained": trained_scenario,
            "improvement": {
                "avg_reward_lift": round(
                    trained_scenario["avg_total_reward"] - baseline_scenario["avg_total_reward"], 4
                ),
                "survival_rate_lift": round(
                    trained_scenario["survival_rate"] - baseline_scenario["survival_rate"], 4
                ),
                "verifier_pass_rate_lift": round(
                    trained_scenario["verifier_pass_rate"] - baseline_scenario["verifier_pass_rate"], 4
                ),
            },
        }

    results["scenario_results"] = scenario_results

    output_dir = Path("demo_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "demo_results.json").open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    save_artifacts(artifacts, str(output_dir / "trained_policy_qtable.json"))

    return results


if __name__ == "__main__":
    demo = run_demo()
    print("=== Startup Survival Simulator Demo ===")
    print("Baseline metrics:", demo["baseline_metrics"])
    print("Trained metrics:", demo["trained_metrics"])
    print("Improvement:", demo["improvement"])
    print("Replay seed:", demo["same_seed_replay"]["seed"])
    print("Results saved to: demo_outputs/demo_results.json")
