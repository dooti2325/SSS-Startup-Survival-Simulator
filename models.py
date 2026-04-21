"""Typed models for the Startup Survival Simulator."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

MIN_PUBLIC_SCORE = 0.001
MAX_PUBLIC_SCORE = 0.999


class Action(str, Enum):
    """Supported actions for each environment step."""

    INCREASE_MARKETING = "increase_marketing"
    HIRE_ENGINEER = "hire_engineer"
    IMPROVE_PRODUCT = "improve_product"
    REDUCE_COSTS = "reduce_costs"
    PIVOT_MARKET = "pivot_market"
    RAISE_FUNDING = "raise_funding"
    ANALYZE_MARKET = "analyze_market"
    REFACTOR_CODE = "refactor_code"
    DO_NOTHING = "do_nothing"


class StartupState(BaseModel):
    """Observable startup state returned by the environment."""

    cash: float = Field(default=50_000.0, ge=0.0, description="Available cash in USD")
    users: int = Field(default=100, ge=0, description="Current active users")
    revenue: float = Field(default=1_000.0, ge=0.0, description="Current revenue in USD")
    growth_rate: float = Field(default=0.08, ge=0.0, le=1.0, description="Growth multiplier per step")
    burn_rate: float = Field(default=4_500.0, ge=0.0, description="Operating burn per step in USD")
    product_quality: float = Field(default=0.55, ge=0.0, le=1.0, description="Product quality score")
    morale: float = Field(default=0.7, ge=0.0, le=1.0, description="Team morale score")
    time_step: int = Field(default=0, ge=0, description="Environment time step")


class StepResult(BaseModel):
    """Structured output returned from env.step()."""

    state: StartupState
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    """Optional request payload for resetting the environment."""

    seed: Optional[int] = Field(default=None, description="Override environment RNG seed")


class StepRequest(BaseModel):
    """Action payload for stepping the environment."""

    action: Action


class GraderResponse(BaseModel):
    """Score returned by the task grader."""

    score: float = Field(gt=0.0, lt=1.0)

    @field_validator("score")
    @classmethod
    def clamp_score(cls, value: float) -> float:
        """Defend against floating-point drift while keeping scores strictly in (0, 1)."""
        return max(MIN_PUBLIC_SCORE, min(MAX_PUBLIC_SCORE, float(value)))
