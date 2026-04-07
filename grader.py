"""Task graders for the Startup Survival Simulator."""

from models import GraderResponse, StartupState

MIN_SCORE = 1e-6
MAX_SCORE = 1.0 - MIN_SCORE


def _clamp(value: float) -> float:
    """Clamp grader outputs into the validator-safe open interval (0, 1)."""
    return max(MIN_SCORE, min(MAX_SCORE, float(value)))


def grade(task_name: str, state: dict) -> dict:
    """Grade a state snapshot for the requested task."""
    current = StartupState(**state)

    if task_name == "survival":
        progress = current.time_step / 30.0
        cash_penalty = 0.0 if current.cash >= 5_000 else 0.2
        score = _clamp(progress - cash_penalty)
    elif task_name == "growth":
        user_progress = current.users / 1_000.0
        sustainability_bonus = 0.1 if current.burn_rate <= max(current.revenue * 1.2, 1.0) else 0.0
        score = _clamp(user_progress + sustainability_bonus)
    elif task_name == "scaling":
        efficiency = current.revenue / max(current.burn_rate, 1.0)
        user_factor = current.users / 2_000.0
        score = _clamp((efficiency * 0.7) + (user_factor * 0.3))
    else:
        raise ValueError(f"Unknown task: {task_name}")

    return GraderResponse(score=score).model_dump()
