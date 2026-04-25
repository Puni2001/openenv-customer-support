# Final Submission Checklist

Use this right before final push/submission.

## Core Validation
- [ ] `venv/bin/openenv validate .` passes
- [ ] `venv/bin/python test_env.py` passes
- [ ] `venv/bin/python -m server.app` starts cleanly
- [ ] `venv/bin/python test_api.py` passes while server is running

## Required Hackathon Artifacts
- [ ] Hugging Face Space URL is live and in `README.md`
- [ ] Video/blog/slides links are in `README.md`
- [ ] Training script is present and runnable (`train.py`)
- [ ] Colab notebook is present (`train_colab.ipynb`)
- [ ] Reward curves are committed as image files

## Evidence Package
- [ ] `results/final_baseline_vs_trained.md`
- [ ] `results/final_baseline_vs_trained.json`
- [ ] `results/final_summary_stats.json`
- [ ] `results/final_reward_comparison.png`
- [ ] `results/ablation_hack_penalty.json`
- [ ] `results/scorecard_report.json`
- [ ] `results/frontier_scorecard_smoke.md`

## Final Run Refresh (if rerunning with HF credits)
- [ ] `python train.py --model Qwen/Qwen2.5-0.5B-Instruct --curriculum easy,medium,hard --epochs 2 --samples 200 --batch-size 2 --seed 42`
- [ ] `python evaluate_models.py --base-model Qwen/Qwen2.5-0.5B-Instruct --trained-model punith2001/openenv-customer-support-model --tasks easy,medium,hard --episodes 3 --seeds 41,42,43 --output results/final_baseline_vs_trained.md`
- [ ] `python evaluate_models.py --offline --tasks easy,medium,hard,frontier --episodes 3 --seeds 41,42,43 --output results/final_baseline_vs_trained.md --trained-model offline_stub`
- [ ] `python ablation_eval.py`
- [ ] `curl -s http://localhost:7860/export/scorecard > /tmp/scorecard_export.json`
- [ ] README tables refreshed from latest `results/final_*`

## Readme Quality
- [ ] Problem -> Environment -> Reward -> Results flow is clear
- [ ] Metrics in top section match artifact JSON files
- [ ] Ablation evidence is explicitly explained in one paragraph
- [ ] Frontier governance stack (tool evidence + safe handoff + legal hold) is clearly documented
- [ ] Run instructions are copy/paste runnable in order

## Repo Hygiene
- [ ] No large transient outputs (checkpoints) included
- [ ] No secrets committed (especially `.env`)
- [ ] `git status` shows only intended files
- [ ] Final commit message reflects trained evidence + ablation

## Freeze
- [ ] Final links verified one last time
- [ ] Final push done before deadline
- [ ] No further metric/claim edits after submission
