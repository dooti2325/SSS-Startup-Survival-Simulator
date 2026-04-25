# OpenEnv Hackathon Submission Checklist

Use this as a final pass before submission freeze.

## Minimum Requirements

- [x] OpenEnv-compliant API available (`reset`, `step`, `state`) with `openenv.yaml` manifest.
- [x] Hugging Face Space live and linked in README.
- [x] TRL-based training script included (`train_trl.py` — CPU, distilgpt2).
- [x] Unsloth training notebook included (`train_trl.ipynb` — Qwen2.5-7B, T4 Colab).
- [x] Training artifacts committed (`artifacts/training_loss.png`, `evaluation_comparison.png`, `evaluation_summary.json`).
- [x] README has no `TODO` placeholders for required links.
- [x] Submission materials section in README complete.
- [ ] Mini-blog published on HuggingFace — add URL to README once live.
- [ ] Demo video (<2 min) uploaded to YouTube — add URL to README once live.
- [ ] Run `openenv validate` locally and confirm PASS.

## Judging Criteria Self-Score

| Criterion | Weight | Self-Assessment |
|---|---|---|
| Environment Innovation | 40% | ✅ Partial observability, hidden tech debt, sparse rewards, 9 actions, 3 tasks |
| Storytelling | 30% | ✅ README narrative + plots embedded; pending blog/video |
| Showing Improvement in Rewards | 20% | ✅ +0.425 avg score delta, +1000 reward delta committed in artifacts/ |
| Reward & Training Pipeline | 10% | ✅ SFTTrainer + oracle dataset + eval comparison |

## Key Links

- **HF Space:** https://huggingface.co/spaces/Loosebag/SSS-Startup-Survival-Simulator
- **GitHub:** https://github.com/dooti2325/SSS-Startup-Survival-Simulator
- **Colab:** https://colab.research.google.com/github/dooti2325/SSS-Startup-Survival-Simulator/blob/main/train_trl.ipynb
