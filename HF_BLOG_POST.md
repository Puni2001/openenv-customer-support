# FrontierOps Arena: Mini-Blog for Judges

**Demo video (<2 min):** [YouTube demo](https://youtu.be/1-6ji4GL8bc)  
**Live environment:** [Hugging Face Space](https://huggingface.co/spaces/punith2001/frontierops-arena)  
**Submission report:** [HF Discussion #2](https://huggingface.co/spaces/punith2001/frontierops-arena/discussions/2)  
**README (all links):** [GitHub README](https://github.com/Puni2001/frontierops-arena)

## What this environment does

FrontierOps Arena is an OpenEnv benchmark for training enterprise support agents under:
- partial observability
- SLA pressure and queue dynamics
- degraded tools/APIs (timeouts, rate limits, fallback)
- governance constraints in high-risk cases

The core objective is to train policy behavior that is both **effective and safe**, not just fluent.

## What we trained

We trained a policy with a curriculum using GRPO/TRL:
1. `easy` — triage classification
2. `medium` — priority reasoning under SLA/VIP constraints
3. `hard` — resolution quality + escalation logic
4. evaluated on `frontier` stress mode for safe autonomy behavior

Training assets are in `train.py` and `frontierops_training_lab.ipynb`.

## Theme alignment

- **Primary:** Theme #3.1 World Modeling (professional tasks)
- **Secondary:** Theme #1 Multi-Agent Interactions

Theme #1 is implemented via `multi_agent_triage` + `multi_agent_resolver` with incentive-shaped coordination terms (routing/handoff bonuses and anti-dumping penalties), not only role splitting.

## Evidence of improvement

Primary run (`results/baseline_vs_trained_colab.json`):
- Easy: `+56%`
- Medium: `+1,872%`
- Hard: `+284%`

Deterministic repro pack (`results/final_baseline_vs_trained.md`):
- Easy: `+212.0%`
- Medium: `+163.3%`
- Hard: `+81.4%`
- Frontier: `+78.8%`

Reward-hacking ablation (`results/ablation_hack_penalty.json`):
- Removing urgent-spam defense increases exploit reward by `+0.92` (proof safeguard matters)

## Proof Bundle (direct artifacts)

### Core metrics and reproducibility artifacts
- Primary training deltas: [`results/baseline_vs_trained_colab.json`](results/baseline_vs_trained_colab.json)
- Deterministic multi-seed report: [`results/final_baseline_vs_trained.md`](results/final_baseline_vs_trained.md)
- Deterministic raw data: [`results/final_baseline_vs_trained.json`](results/final_baseline_vs_trained.json)
- Summary statistics: [`results/final_summary_stats.json`](results/final_summary_stats.json)
- Judge one-page scorecard: [`results/judge_scorecard.md`](results/judge_scorecard.md)
- Judge Q&A: [`JUDGE_QA.md`](JUDGE_QA.md)

### Safety, governance, and anti-reward-hacking proof
- Anti-hacking ablation: [`results/ablation_hack_penalty.json`](results/ablation_hack_penalty.json)
- Scorecard export output: [`results/scorecard_report.json`](results/scorecard_report.json)
- Frontier scorecard smoke sample: [`results/frontier_scorecard_smoke.json`](results/frontier_scorecard_smoke.json)

### Visual proof (plots)
- Reward curves: [`results/reward_curves.png`](results/reward_curves.png)
- Final reward comparison plot: [`results/final_reward_comparison.png`](results/final_reward_comparison.png)

### Environment and training code proof
- OpenEnv manifest: [`openenv.yaml`](openenv.yaml)
- Environment core: [`src/customer_support_env.py`](src/customer_support_env.py)
- Training script: [`train.py`](train.py)
- Colab training notebook: [`frontierops_training_lab.ipynb`](frontierops_training_lab.ipynb)
- Evaluation script: [`evaluate_models.py`](evaluate_models.py)
- Ablation script: [`ablation_eval.py`](ablation_eval.py)

## Frontier interpretation (important)

Frontier mode is a **safety-constrained stress test**:
- high-risk governance gates
- noisy multilingual/code-mix inputs
- degraded tool reliability

So we treat frontier as safe-autonomy evaluation, not a simple score-max track.  
Low raw success can still represent improvement when unsafe autonomy is reduced and safe handoffs increase.

## Reproducibility path

```bash
python evaluate_models.py --offline --tasks easy,medium,hard,frontier --episodes 3 --seeds 41,42,43 --output results/final_baseline_vs_trained.md --trained-model offline_stub
python ablation_eval.py
python -m server.app   # then GET /export/scorecard
```

## Judge fast path

- `results/judge_scorecard.md` — one-page capability/safety/reliability summary
- `JUDGE_QA.md` — concise technical Q&A
- Space routes: `/` and `/history` for interactive demo + evidence trail

---
Built for the Meta PyTorch OpenEnv Hackathon 2026 Grand Finale.
