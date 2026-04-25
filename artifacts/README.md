# Artifacts Folder

`train_trl.py` writes evaluation evidence here.

Expected files after a successful run:

- `evaluation_summary.json`
- `training_log_history.json`
- `training_loss.png`
- `evaluation_comparison.png`

Heavy checkpoint files are kept under `artifacts/trl_model/` and are git-ignored by default.
