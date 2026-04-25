"""Smoke tests for hackathon-specific SSS environment and training flow."""

from sss_hackathon_env import ACTIONS, StartupSurvivalEnv
from sss_reward_verifier import verify_episode
from sss_training import (
    build_greedy_policy,
    evaluate_policy,
    evaluate_policy_in_scenario,
    train_q_learning,
)


def test_required_actions_and_state_fields() -> None:
    env = StartupSurvivalEnv(seed=42)
    state = env.reset(seed=42)

    assert ACTIONS == ["hire", "fire", "build_feature", "pivot", "marketing_spend", "do_nothing"]
    for field in ("funding", "team_size", "burn_rate", "market_demand", "runway"):
        assert field in state


def test_step_and_verifier_execute() -> None:
    env = StartupSurvivalEnv(seed=42)
    env.reset(seed=42)
    trajectory = []
    done = False
    while not done:
        out = env.step("do_nothing")
        trajectory.append(out)
        done = out["done"]

    verdict = verify_episode(trajectory)
    assert "checks" in verdict
    assert "survived_x_steps" in verdict["checks"]


def test_training_improves_survival_rate() -> None:
    seeds = [7, 11, 19, 23, 31, 43]
    random_baseline = evaluate_policy(lambda _: "do_nothing", seeds)
    artifacts = train_q_learning(episodes=120, seed=2026)
    trained = evaluate_policy(build_greedy_policy(artifacts), seeds)
    assert trained["survival_rate"] >= random_baseline["survival_rate"]


def test_scenarios_change_dynamics_for_same_seed() -> None:
    standard_env = StartupSurvivalEnv(seed=42, scenario="standard")
    recession_env = StartupSurvivalEnv(seed=42, scenario="recession")

    standard_env.reset(seed=42)
    recession_env.reset(seed=42)

    standard_step = standard_env.step("do_nothing")
    recession_step = recession_env.step("do_nothing")

    assert standard_step["info"]["scenario"] == "standard"
    assert recession_step["info"]["scenario"] == "recession"
    assert recession_step["state"]["market_demand"] <= standard_step["state"]["market_demand"]


def test_evaluate_policy_in_scenario_shape() -> None:
    seeds = [7, 11, 19]
    metrics = evaluate_policy_in_scenario(lambda _: "do_nothing", seeds, scenario="competition")
    assert metrics["scenario"] == "competition"
    assert {"avg_total_reward", "survival_rate", "verifier_pass_rate"} <= metrics.keys()
