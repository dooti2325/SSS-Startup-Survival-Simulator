"""Systematic stress/debug checks for Startup Survival Simulator.

Checks performed:
1) Edge-case action stress tests
2) Reward hacking detection
3) Reward-to-quality alignment validation
4) Trivial strategy guardrail (do-nothing should not score high)
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List

from sss_hackathon_env import ACTIONS
from sss_training import (
    build_greedy_policy,
    build_random_policy,
    evaluate_policy,
    run_episode,
    train_q_learning,
)
from sss_reward_verifier import verify_episode


def _fixed_action_policy(action: str) -> Callable[[Dict[str, float]], str]:
    def _policy(_: Dict[str, float]) -> str:
        return action

    return _policy


def run_stress_debug(seeds: List[int] | None = None) -> Dict[str, object]:
    if seeds is None:
        seeds = list(range(1, 41))

    policies: Dict[str, Callable[[Dict[str, float]], str]] = {
        f"always_{action}": _fixed_action_policy(action) for action in ACTIONS
    }
    policies["random"] = build_random_policy(seed=2026)

    artifacts = train_q_learning(episodes=350, seed=2026)
    policies["trained_q"] = build_greedy_policy(artifacts)

    metrics = {name: evaluate_policy(policy, seeds) for name, policy in policies.items()}

    # Reward alignment: correlation between total reward and a startup-quality proxy.
    rewards: List[float] = []
    quality_proxy: List[float] = []
    suspicious = []

    # Reuse run_episode to avoid alternate code paths.
    from sss_hackathon_env import StartupSurvivalEnv

    env = StartupSurvivalEnv(seed=999)
    for seed in seeds:
        total_reward, trajectory = run_episode(env, policies["trained_q"], seed=seed)
        verdict = verify_episode(trajectory)
        final_state = trajectory[-1]["state"]

        proxy = (
            (1 if final_state["funding"] > 0 else 0)
            + (1 if final_state["revenue"] > final_state["burn_rate"] else 0)
            + (final_state["runway"] / 20.0)
            - (final_state["constraint_violations"] * 0.5)
        )
        rewards.append(total_reward)
        quality_proxy.append(proxy)

        if total_reward > -20 and not verdict["passed"]:
            suspicious.append(
                {
                    "seed": seed,
                    "reward": round(total_reward, 4),
                    "checks": verdict["checks"],
                    "final_state": final_state,
                }
            )

    mx = sum(rewards) / len(rewards)
    my = sum(quality_proxy) / len(quality_proxy)
    cov = sum((x - mx) * (y - my) for x, y in zip(rewards, quality_proxy))
    sdx = math.sqrt(sum((x - mx) ** 2 for x in rewards))
    sdy = math.sqrt(sum((y - my) ** 2 for y in quality_proxy))
    reward_alignment_corr = cov / (sdx * sdy) if sdx and sdy else 0.0

    do_nothing_reward = metrics["always_do_nothing"]["avg_total_reward"]
    trained_reward = metrics["trained_q"]["avg_total_reward"]
    random_reward = metrics["random"]["avg_total_reward"]

    assertions = {
        "no_trivial_high_reward": bool(do_nothing_reward < random_reward and do_nothing_reward < trained_reward),
        "trained_beats_random_reward": bool(trained_reward > random_reward),
        "trained_beats_random_verifier": bool(
            metrics["trained_q"]["verifier_pass_rate"] >= metrics["random"]["verifier_pass_rate"]
        ),
        "reward_alignment_positive": bool(reward_alignment_corr > 0.5),
        "no_high_reward_failed_verifier_cases": bool(len(suspicious) == 0),
    }

    return {
        "seeds": seeds,
        "metrics": metrics,
        "reward_alignment_corr": round(reward_alignment_corr, 4),
        "suspicious_cases": suspicious,
        "assertions": assertions,
        "all_assertions_passed": all(assertions.values()),
    }


if __name__ == "__main__":
    report = run_stress_debug()
    print("=== SSS Stress Debug Report ===")
    print("All assertions passed:", report["all_assertions_passed"])
    print("Assertions:", report["assertions"])
    print("Reward alignment correlation:", report["reward_alignment_corr"])
    print("Key metrics:")
    for name, metric in report["metrics"].items():
        print(name, metric)
    print("Suspicious high-reward failure cases:", len(report["suspicious_cases"]))
