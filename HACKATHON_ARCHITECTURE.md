# Startup Survival Simulator (SSS) - Hackathon Architecture

## Phase 1: System Design

SSS is a deterministic, seed-driven startup simulation environment where an agent selects one action per step and receives a layered reward. The system is optimized for verifiable improvement and fast demo execution.

Core loop:
1. `reset(seed)` initializes a startup state.
2. Agent chooses action from discrete action space.
3. `step(action)` updates state under constraints.
4. Reward is computed with growth/survival/penalty layers.
5. Episode ends on bankruptcy or max steps.
6. Verifier checks deterministic success conditions.

## Module Breakdown

- `sss_hackathon_env.py`
  - OpenEnv-style environment (`reset`, `step`, `state`)
  - State transitions, anti-cheat constraints, deterministic market dynamics
- `sss_reward_verifier.py`
  - Layered reward function
  - Deterministic verification checks
- `sss_training.py`
  - Tabular Q-learning loop
  - Random baseline and trained policy evaluation
- `sss_demo.py`
  - End-to-end reproducible baseline vs trained comparison
  - Same-seed replay and JSON artifact generation

## Data Flow

1. `sss_demo.py` starts training/evaluation.
2. `sss_training.py` calls `StartupSurvivalEnv.reset/step`.
3. `sss_hackathon_env.py` updates state and calls `compute_reward`.
4. Trajectories are passed to `verify_episode`.
5. Metrics are aggregated and written to `demo_outputs/demo_results.json`.

## Key Verifiability Hooks

- Fixed scenario seeds for baseline and trained policies.
- Same-seed replay for side-by-side behavioral comparison.
- Deterministic checks:
  - survived required steps
  - revenue > burn
  - zero constraint violations
