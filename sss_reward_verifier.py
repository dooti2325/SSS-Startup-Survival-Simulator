"""Reward logic and deterministic verification for the hackathon simulator."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List


def compute_reward(previous_state, current_state, info: Dict[str, object]) -> float:
    """Layered reward with anti-cheat penalties.

    Components:
    - positive: revenue growth + sustainable runway
    - negative: burn pressure + bad decisions + time penalty
    """
    revenue_delta = current_state.revenue - previous_state.revenue
    burn_penalty = current_state.burn_rate * 0.0016
    survival_bonus = 1.4 if current_state.funding > 0 else -4.0
    runway_bonus = 0.0
    if current_state.runway >= 12:
        runway_bonus = 1.2
    elif current_state.runway >= 6:
        runway_bonus = 0.6
    else:
        runway_bonus = -0.8

    growth_signal = revenue_delta * 0.011
    bad_decision_penalty = float(info.get("bad_decision_penalty", 0.0)) * 1.8
    constraint_penalty = current_state.constraint_violations * 4.0
    time_penalty = 0.20  # discourages "stalling forever"

    reward = (
        growth_signal
        + survival_bonus
        + runway_bonus
        - burn_penalty
        - bad_decision_penalty
        - constraint_penalty
        - time_penalty
    )
    return round(float(reward), 4)


def verify_episode(
    trajectory: List[Dict[str, object]],
    required_survival_steps: int = 30,
) -> Dict[str, object]:
    """Deterministic checks requested by hackathon brief."""
    if not trajectory:
        return {
            "passed": False,
            "checks": {
                "survived_x_steps": False,
                "revenue_gt_burn": False,
                "no_constraint_violations": False,
            },
            "summary": "Empty trajectory.",
        }

    final_state = trajectory[-1]["state"]
    survived_x_steps = bool(final_state["time_step"] >= required_survival_steps and final_state["funding"] > 0.0)
    revenue_gt_burn = bool(final_state["revenue"] > final_state["burn_rate"])
    no_constraint_violations = bool(final_state["constraint_violations"] == 0)

    checks = {
        "survived_x_steps": survived_x_steps,
        "revenue_gt_burn": revenue_gt_burn,
        "no_constraint_violations": no_constraint_violations,
    }
    passed = all(checks.values())
    summary = "PASS" if passed else "FAIL"

    return {
        "passed": passed,
        "checks": checks,
        "summary": summary,
        "final_state": final_state,
    }


def compact_state_for_logging(state_obj) -> Dict[str, float]:
    """Utility helper for clean logs."""
    if hasattr(state_obj, "__dict__"):
        return {k: v for k, v in asdict(state_obj).items()}
    return dict(state_obj)
