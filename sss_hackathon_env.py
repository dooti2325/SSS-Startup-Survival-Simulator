"""Hackathon-focused Startup Survival Simulator environment.

OpenEnv-style API:
- reset()
- step(action)
- state representation
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from sss_reward_verifier import compute_reward


ACTIONS: List[str] = [
    "hire",
    "fire",
    "build_feature",
    "pivot",
    "marketing_spend",
    "do_nothing",
]

SCENARIOS: Dict[str, Dict[str, float]] = {
    "standard": {
        "market_demand_shift": 0.0,
        "revenue_multiplier": 1.0,
        "burn_multiplier": 1.0,
        "product_decay": 0.0,
    },
    "recession": {
        "market_demand_shift": -0.015,
        "revenue_multiplier": 0.9,
        "burn_multiplier": 1.03,
        "product_decay": 0.0,
    },
    "competition": {
        "market_demand_shift": -0.008,
        "revenue_multiplier": 0.94,
        "burn_multiplier": 1.01,
        "product_decay": 0.003,
    },
}


@dataclass
class StartupState:
    """Visible simulation state (includes mandatory hackathon fields)."""

    funding: float = 50_000.0
    team_size: int = 4
    burn_rate: float = 4_500.0
    market_demand: float = 0.55
    runway: float = 11.11
    revenue: float = 2_200.0
    product_strength: float = 0.45
    time_step: int = 0
    constraint_violations: int = 0


class StartupSurvivalEnv:
    """Deterministic startup management simulator with anti-cheat constraints."""

    def __init__(self, seed: int = 42, max_steps: int = 40, scenario: str = "standard") -> None:
        self.base_seed = seed
        self.max_steps = max_steps
        self.max_team_size = 16
        self.scenario = scenario
        self.rng = random.Random(seed)
        self.current_state = StartupState()
        self.reset(seed=seed, scenario=scenario)

    def reset(self, seed: Optional[int] = None, scenario: Optional[str] = None) -> Dict[str, float]:
        """Reset environment to initial startup configuration."""
        if seed is not None:
            self.base_seed = seed
        if scenario is not None:
            if scenario not in SCENARIOS:
                raise ValueError(f"Invalid scenario: {scenario}")
            self.scenario = scenario
        self.rng = random.Random(self.base_seed)
        self.current_state = StartupState()
        self._update_runway()
        return self.state()

    def state(self) -> Dict[str, float]:
        """Return current state as a serializable dict."""
        return asdict(self.current_state)

    def step(self, action: str) -> Dict[str, object]:
        """Advance simulation by one step."""
        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        previous_state = StartupState(**self.state())
        info: Dict[str, object] = {"action": action, "bad_decision_penalty": 0.0}
        self._apply_action(action, info)
        if self.current_state.constraint_violations > previous_state.constraint_violations:
            self._apply_violation_penalty()
        self._advance_market()
        self._advance_financials()

        self.current_state.time_step += 1
        self._update_runway()
        reward = compute_reward(previous_state=previous_state, current_state=self.current_state, info=info)
        done, reason = self._is_done()
        info["reason"] = reason
        info["scenario"] = self.scenario

        return {
            "state": self.state(),
            "reward": reward,
            "done": done,
            "info": info,
        }

    def _spend(self, amount: float, info: Dict[str, object], penalty: float = 2.0) -> bool:
        """Spend funding if possible; otherwise apply constraint violation penalty."""
        if self.current_state.funding < amount:
            self.current_state.constraint_violations += 1
            info["bad_decision_penalty"] = float(info.get("bad_decision_penalty", 0.0)) + penalty
            info["constraint_violation"] = True
            return False
        self.current_state.funding -= amount
        return True

    def _apply_action(self, action: str, info: Dict[str, object]) -> None:
        st = self.current_state

        if action == "hire":
            if st.team_size >= self.max_team_size:
                st.constraint_violations += 1
                info["bad_decision_penalty"] = float(info.get("bad_decision_penalty", 0.0)) + 1.5
                info["constraint_violation"] = True
                return
            if not self._spend(2_000.0, info):
                return
            st.team_size += 1
            st.burn_rate += 850.0
            st.product_strength = min(1.0, st.product_strength + 0.03)

        elif action == "fire":
            if st.team_size <= 1:
                st.constraint_violations += 1
                info["bad_decision_penalty"] = float(info.get("bad_decision_penalty", 0.0)) + 1.0
                info["constraint_violation"] = True
                return
            st.team_size -= 1
            st.burn_rate = max(1_400.0, st.burn_rate - 800.0)
            st.product_strength = max(0.1, st.product_strength - 0.015)

        elif action == "build_feature":
            if not self._spend(1_100.0, info):
                return
            quality_gain = 0.025 + (0.002 * min(st.team_size, 10))
            st.product_strength = min(1.0, st.product_strength + quality_gain)

        elif action == "pivot":
            if not self._spend(1_500.0, info):
                return
            # Risky but potentially high upside.
            demand_shift = self.rng.uniform(-0.18, 0.24)
            strength_shift = self.rng.uniform(-0.08, 0.12)
            st.market_demand = min(1.0, max(0.05, st.market_demand + demand_shift))
            st.product_strength = min(1.0, max(0.05, st.product_strength + strength_shift))
            info["pivot_shift"] = round(demand_shift, 4)

        elif action == "marketing_spend":
            if not self._spend(1_000.0, info):
                return
            immediate_boost = self.rng.uniform(0.01, 0.06)
            st.market_demand = min(1.0, st.market_demand + immediate_boost)
            st.burn_rate += 150.0

        elif action == "do_nothing":
            # Passive decay prevents "wait forever" exploits.
            st.market_demand = max(0.05, st.market_demand - 0.01)
            st.product_strength = max(0.05, st.product_strength - 0.004)

    def _advance_market(self) -> None:
        st = self.current_state
        market_noise = self.rng.uniform(-0.03, 0.03)
        scenario_shift = SCENARIOS[self.scenario]["market_demand_shift"]
        st.market_demand = min(1.0, max(0.05, st.market_demand + market_noise + scenario_shift))
        decay = SCENARIOS[self.scenario]["product_decay"]
        if decay > 0:
            st.product_strength = max(0.05, st.product_strength - decay)

    def _apply_violation_penalty(self) -> None:
        """Immediate hard penalty for invalid decisions to prevent reward hacking."""
        st = self.current_state
        st.funding = max(0.0, st.funding - 1_600.0)
        st.revenue = max(0.0, st.revenue * 0.95)
        st.product_strength = max(0.05, st.product_strength - 0.01)

    def _advance_financials(self) -> None:
        st = self.current_state
        # Diminishing returns enforce anti-infinite-scaling behavior.
        team_factor = min(1.8, 0.55 + (st.team_size * 0.09))
        saturation = max(0.35, 1.0 - (st.revenue / 150_000.0))
        growth_multiplier = team_factor * st.product_strength * st.market_demand * saturation
        revenue_gain = max(0.0, 700.0 * growth_multiplier + self.rng.uniform(-120.0, 180.0))
        revenue_gain *= SCENARIOS[self.scenario]["revenue_multiplier"]
        st.revenue = max(0.0, st.revenue + revenue_gain)

        st.funding = max(0.0, st.funding + st.revenue - st.burn_rate)
        next_burn = max(1_200.0, st.burn_rate + self.rng.uniform(-60.0, 90.0))
        st.burn_rate = next_burn * SCENARIOS[self.scenario]["burn_multiplier"]

    def _update_runway(self) -> None:
        st = self.current_state
        st.runway = round(st.funding / max(st.burn_rate, 1.0), 2)

    def _is_done(self) -> Tuple[bool, str]:
        st = self.current_state
        if st.funding <= 0.0:
            return True, "bankrupt"
        if st.constraint_violations >= 3:
            return True, "constraint_breach"
        if st.time_step >= self.max_steps:
            return True, "max_steps_reached"
        return False, "ongoing"
