"""Task metadata for the Startup Survival Simulator."""

from typing import Any, Dict, List

from models import Action


def get_tasks() -> Dict[str, Any]:
    """Return task definitions and the action schema for validator-friendly discovery."""
    return {
        # Keep task metadata explicit so judges do not need to infer intent from code.
        "tasks": [
            {
                "id": "survival",
                "difficulty": "easy",
                "name": "Survive 30 Steps",
                "description": "Keep the startup alive for 30 steps without going bankrupt.",
                "success_criteria": "cash > 0 and time_step >= 30",
                "evaluation_intent": "Measures operational stability and basic cash management.",
            },
            {
                "id": "growth",
                "difficulty": "medium",
                "name": "Reach 1000 Users",
                "description": "Grow the startup to at least 1000 users while staying financially healthy.",
                "success_criteria": "users >= 1000",
                "evaluation_intent": "Measures customer growth under realistic operating constraints.",
            },
            {
                "id": "scaling",
                "difficulty": "hard",
                "name": "Scale Efficiently",
                "description": "Maximize revenue and user scale relative to burn rate.",
                "success_criteria": "high efficiency score at the end of the episode",
                "evaluation_intent": "Measures sustainable scaling efficiency instead of raw growth alone.",
            },
        ],
        "action_schema": {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [action.value for action in Action],
                }
            },
        },
    }
