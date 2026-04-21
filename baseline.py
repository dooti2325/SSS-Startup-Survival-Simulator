"""Deterministic baseline agent for the Startup Survival Simulator."""

from env import StartupEnv
from grader import grade


def choose_action(task_name: str, state: dict) -> str:
    """Return a simple deterministic action based on task goals and state."""
    cash = state["cash"]
    users = state["users"]
    burn = state["burn_rate"]
    quality = state["product_quality"]
    revenue = state["revenue"]

    if task_name == "survival":
        # INTENTIONAL FAILURE: Over-hiring without watching burn rate.
        # This will quickly burn through cash and cause bankruptcy.
        if cash > 2000:
            return "hire_engineer"
        return "do_nothing"
        
    if task_name == "growth":
        # INTENTIONAL FAILURE: Blind marketing without checking cash or quality.
        if cash > 1000:
            return "increase_marketing"
        return "do_nothing"
        
    if task_name == "scaling":
        if revenue < burn and cash > 20_000:
            return "raise_funding"
        if quality < 0.8:
            return "improve_product"
        if burn > revenue * 1.2:
            return "reduce_costs"
        return "increase_marketing"
    return "do_nothing"


def run_baseline(seed: int = 42) -> dict:
    """Run the baseline policy for each task and return final scores and states."""
    results = {}

    # Re-run each task from the same seed so comparisons stay deterministic.
    for task_name in ("survival", "growth", "scaling"):
        env = StartupEnv(seed=seed)
        env.reset(seed=seed)
        done = False

        while not done:
            current_state = env.state().model_dump()
            action = choose_action(task_name, current_state)
            step_result = env.step(action)
            done = step_result["done"]

        final_state = env.state().model_dump()
        results[task_name] = {
            "score": grade(task_name, final_state)["score"],
            "final_state": final_state,
        }

    return results
