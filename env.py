"""Core environment logic for the Startup Survival Simulator."""

import random
from typing import Dict, Optional

from models import Action, StartupState, StepResult


class StartupEnv:
    """A lightweight environment for startup decision making."""

    def __init__(self, seed: Optional[int] = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self._market_demand = 0.6
        self._churn_rate = 0.03
        self.current_state = StartupState()
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> StartupState:
        """Reset the environment to a reproducible startup starting point."""
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self._market_demand = 0.6
        self._churn_rate = 0.03
        self.current_state = StartupState()
        return self.current_state

    def state(self) -> StartupState:
        """Return the current observable state."""
        return self.current_state

    def step(self, action: str) -> Dict[str, object]:
        """Apply an action, advance the simulation, and return the step result."""
        try:
            selected_action = Action(action)
        except ValueError as exc:
            raise ValueError(f"Invalid action: {action}") from exc

        state_before = self.current_state.model_copy(deep=True)
        state = self.current_state
        info: Dict[str, object] = {"action": selected_action.value}

        self._apply_action_effects(state, selected_action, info)
        self._apply_market_noise(state)

        prev_users = state.users
        prev_revenue = state.revenue

        # Growth is applied before revenue so monetization reflects the updated user base.
        acquired_users = self._calculate_new_users(state)
        lost_users = min(state.users, int(state.users * self._churn_rate))
        state.users = max(0, state.users + acquired_users - lost_users)

        arpu = 14.0 + (state.product_quality * 6.0)
        state.revenue = round(state.users * arpu * max(0.35, self._market_demand), 2)
        state.cash = round(max(0.0, state.cash + state.revenue - state.burn_rate), 2)
        state.time_step += 1

        reward = self._calculate_reward(
            state_before=state_before,
            acquired_users=acquired_users,
            lost_users=lost_users,
            revenue_delta=state.revenue - prev_revenue,
        )
        done, reason = self._check_done(state)
        if reason:
            info["reason"] = reason
        info["acquired_users"] = acquired_users
        info["lost_users"] = lost_users
        info["net_users"] = state.users - prev_users
        info["revenue_delta"] = round(state.revenue - prev_revenue, 2)

        return StepResult(state=state, reward=reward, done=done, info=info).model_dump()

    def _apply_action_effects(self, state: StartupState, action: Action, info: Dict[str, object]) -> None:
        if action == Action.INCREASE_MARKETING:
            state.burn_rate += 1_200.0
            state.growth_rate = min(0.35, state.growth_rate + 0.03)
            self._market_demand = min(1.0, self._market_demand + 0.03)
        elif action == Action.HIRE_ENGINEER:
            state.burn_rate += 2_200.0
            state.product_quality = min(1.0, state.product_quality + 0.07)
            state.morale = min(1.0, state.morale + 0.03)
        elif action == Action.IMPROVE_PRODUCT:
            state.product_quality = min(1.0, state.product_quality + 0.05)
            self._churn_rate = max(0.01, self._churn_rate - 0.006)
            state.morale = min(1.0, state.morale + 0.02)
        elif action == Action.REDUCE_COSTS:
            state.burn_rate = max(1_800.0, state.burn_rate - 1_600.0)
            state.growth_rate = max(0.02, state.growth_rate - 0.015)
            state.morale = max(0.2, state.morale - 0.06)
        elif action == Action.PIVOT_MARKET:
            demand_shift = self.rng.uniform(-0.08, 0.15)
            quality_shift = self.rng.uniform(-0.03, 0.04)
            self._market_demand = max(0.2, min(1.0, self._market_demand + demand_shift))
            state.product_quality = max(0.2, min(1.0, state.product_quality + quality_shift))
            self._churn_rate = min(0.2, self._churn_rate + 0.01)
            info["pivot_shift"] = round(demand_shift, 3)
        elif action == Action.RAISE_FUNDING:
            probability = min(0.85, 0.2 + (state.product_quality * 0.3) + min(0.35, state.users / 5000))
            if self.rng.random() < probability:
                state.cash += 30_000.0
                state.burn_rate += 600.0
                info["funding_raised"] = 30_000.0
            else:
                info["funding_raised"] = 0.0
                state.morale = max(0.2, state.morale - 0.03)
        elif action == Action.DO_NOTHING:
            state.morale = max(0.2, state.morale - 0.01)

        state.burn_rate = round(state.burn_rate, 2)
        state.growth_rate = round(max(0.0, min(0.5, state.growth_rate)), 4)
        self._churn_rate = round(max(0.0, min(0.25, self._churn_rate)), 4)
        state.product_quality = round(max(0.0, min(1.0, state.product_quality)), 4)
        self._market_demand = round(max(0.0, min(1.0, self._market_demand)), 4)
        state.morale = round(max(0.0, min(1.0, state.morale)), 4)

    def _apply_market_noise(self, state: StartupState) -> None:
        market_noise = self.rng.uniform(-0.02, 0.02)
        churn_noise = self.rng.uniform(-0.003, 0.003)
        self._market_demand = round(max(0.2, min(1.0, self._market_demand + market_noise)), 4)
        self._churn_rate = round(max(0.01, min(0.25, self._churn_rate + churn_noise)), 4)

    def _calculate_new_users(self, state: StartupState) -> int:
        base_growth = state.users * state.growth_rate
        quality_multiplier = 0.7 + (state.product_quality * 0.6)
        demand_multiplier = 0.6 + (self._market_demand * 0.8)
        morale_multiplier = 0.75 + (state.morale * 0.35)
        bootstrap_users = 25 if state.users < 150 else 0
        new_users = int(base_growth * quality_multiplier * demand_multiplier * morale_multiplier) + bootstrap_users
        return max(0, new_users)

    def _calculate_reward(
        self,
        state_before: StartupState,
        acquired_users: int,
        lost_users: int,
        revenue_delta: float,
    ) -> float:
        """
        Simplified Reward:
        Base reward is the net new users acquired this step.
        If the startup goes bankrupt, it receives a heavy penalty.
        """
        net_users = acquired_users - lost_users
        reward = float(net_users)
        
        # Apply bankruptcy penalty
        if self.current_state.cash <= 0:
            reward -= 100.0
            
        return round(reward, 4)

    def _check_done(self, state: StartupState) -> tuple[bool, Optional[str]]:
        if state.cash <= 0:
            return True, "bankrupt"
        if state.users >= 10_000:
            return True, "success"
        if state.time_step >= 50:
            return True, "timeout"
        return False, None
