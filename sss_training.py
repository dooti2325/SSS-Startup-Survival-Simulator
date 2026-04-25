"""Minimal RL training loop for Startup Survival Simulator."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sss_hackathon_env import ACTIONS, StartupSurvivalEnv
from sss_reward_verifier import verify_episode


@dataclass
class PolicyArtifacts:
    q_table: Dict[Tuple[int, int, int, int, int], List[float]]
    training_rewards: List[float]
    action_names: List[str]


def _discretize(value: float, bins: List[float]) -> int:
    for idx, boundary in enumerate(bins):
        if value <= boundary:
            return idx
    return len(bins)


def state_to_bucket(state: Dict[str, float]) -> Tuple[int, int, int, int, int]:
    """State abstraction for tabular Q-learning."""
    funding_bin = _discretize(state["funding"], [8_000, 20_000, 40_000, 70_000])
    team_bin = _discretize(state["team_size"], [2, 4, 6, 10, 14])
    burn_bin = _discretize(state["burn_rate"], [2_500, 4_500, 6_500, 8_500, 11_000])
    demand_bin = _discretize(state["market_demand"], [0.2, 0.4, 0.6, 0.8])
    runway_bin = _discretize(state["runway"], [2.0, 4.0, 8.0, 12.0, 18.0])
    return (funding_bin, team_bin, burn_bin, demand_bin, runway_bin)


def run_episode(
    env: StartupSurvivalEnv,
    policy_fn,
    seed: int,
) -> Tuple[float, List[Dict[str, object]]]:
    state = env.reset(seed=seed)
    done = False
    trajectory: List[Dict[str, object]] = []
    total_reward = 0.0

    while not done:
        action = policy_fn(state)
        step_out = env.step(action)
        done = bool(step_out["done"])
        total_reward += float(step_out["reward"])
        trajectory.append(step_out)
        state = step_out["state"]
    return total_reward, trajectory


def train_q_learning(
    episodes: int = 350,
    alpha: float = 0.12,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_final: float = 0.08,
    seed: int = 42,
) -> PolicyArtifacts:
    rng = random.Random(seed)
    env = StartupSurvivalEnv(seed=seed)
    q_table: Dict[Tuple[int, int, int, int, int], List[float]] = {}
    training_rewards: List[float] = []

    def ensure_state(bucket):
        if bucket not in q_table:
            q_table[bucket] = [0.0 for _ in ACTIONS]

    for episode in range(episodes):
        progress = episode / max(1, (episodes - 1))
        epsilon = epsilon_start - ((epsilon_start - epsilon_final) * progress)

        state = env.reset(seed=seed + episode)
        done = False
        ep_reward = 0.0

        while not done:
            bucket = state_to_bucket(state)
            ensure_state(bucket)

            if rng.random() < epsilon:
                action_index = rng.randrange(len(ACTIONS))
            else:
                action_index = max(range(len(ACTIONS)), key=lambda i: q_table[bucket][i])

            action = ACTIONS[action_index]
            out = env.step(action)
            next_state = out["state"]
            reward = float(out["reward"])
            done = bool(out["done"])
            ep_reward += reward

            next_bucket = state_to_bucket(next_state)
            ensure_state(next_bucket)
            next_best = max(q_table[next_bucket])
            old_q = q_table[bucket][action_index]
            q_table[bucket][action_index] = old_q + alpha * (reward + (gamma * next_best) - old_q)

            state = next_state

        training_rewards.append(round(ep_reward, 4))

    return PolicyArtifacts(q_table=q_table, training_rewards=training_rewards, action_names=ACTIONS[:])


def build_greedy_policy(artifacts: PolicyArtifacts):
    def policy(state: Dict[str, float]) -> str:
        bucket = state_to_bucket(state)
        q_values = artifacts.q_table.get(bucket)
        if not q_values:
            return "do_nothing"
        best_idx = max(range(len(q_values)), key=lambda i: q_values[i])
        return artifacts.action_names[best_idx]

    return policy


def build_random_policy(seed: int = 123):
    rng = random.Random(seed)

    def policy(_: Dict[str, float]) -> str:
        return ACTIONS[rng.randrange(len(ACTIONS))]

    return policy


def evaluate_policy(policy_fn, seeds: List[int]) -> Dict[str, object]:
    return evaluate_policy_in_scenario(policy_fn, seeds, scenario="standard")


def evaluate_policy_in_scenario(policy_fn, seeds: List[int], scenario: str = "standard") -> Dict[str, object]:
    env = StartupSurvivalEnv(seed=seeds[0] if seeds else 42, scenario=scenario)
    rewards: List[float] = []
    verifier_passes = 0
    survival_passes = 0

    for scenario_seed in seeds:
        total_reward, trajectory = run_episode(env, policy_fn, seed=scenario_seed)
        rewards.append(round(total_reward, 4))
        verdict = verify_episode(trajectory)
        if verdict["passed"]:
            verifier_passes += 1
        if verdict["checks"]["survived_x_steps"]:
            survival_passes += 1

    return {
        "scenario": scenario,
        "scenario_count": len(seeds),
        "avg_total_reward": round(sum(rewards) / max(len(rewards), 1), 4),
        "min_total_reward": min(rewards) if rewards else 0.0,
        "max_total_reward": max(rewards) if rewards else 0.0,
        "verifier_pass_rate": round(verifier_passes / max(len(seeds), 1), 4),
        "survival_rate": round(survival_passes / max(len(seeds), 1), 4),
        "episode_rewards": rewards,
    }


def save_artifacts(artifacts: PolicyArtifacts, output_path: str) -> None:
    serializable = {
        "q_table": {"|".join(map(str, k)): v for k, v in artifacts.q_table.items()},
        "training_rewards": artifacts.training_rewards,
        "action_names": artifacts.action_names,
    }
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(serializable, fp, indent=2)
