---
title: OpenEnv | FrontierOps Arena
emoji: 🎧
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
---

# FrontierOps Arena

**An OpenEnv benchmark for training policy-aware, tool-using enterprise agents under partial observability, delayed rewards, and safety constraints.**

**Live assets**
- Technical blog/report: [Grand Finale Submission Report](https://huggingface.co/spaces/punith2001/frontierops-arena/discussions/2)
- Demo video: [YouTube demo](https://youtu.be/1-6ji4GL8bc)
- Hugging Face Space: [frontierops-arena](https://huggingface.co/spaces/punith2001/frontierops-arena)

[![Watch the Demo](https://img.youtube.com/vi/1-6ji4GL8bc/0.jpg)](https://youtu.be/1-6ji4GL8bc)
*Click above to watch the benchmark in action.*

**Hackathon themes:** #3.1 World Modeling (primary), #1 Multi-Agent Interactions (active), #4 Self-Improvement (roadmap)

---

## Problem and Scope

LLMs often fail in enterprise workflows because they are not trained under uncertainty, conflicting objectives, tool outages, and governance constraints.

This project provides an OpenEnv-compliant benchmark to train and evaluate behavior under those constraints.

---

## Judge Fast Path

1. One-page evidence summary: [`results/judge_scorecard.md`](results/judge_scorecard.md)
2. Q&A and risk handling: [`JUDGE_QA.md`](JUDGE_QA.md)
3. Primary training evidence: [`results/baseline_vs_trained_colab.json`](results/baseline_vs_trained_colab.json), [`results/reward_curves.png`](results/reward_curves.png)
4. Deterministic repro pack: [`results/final_baseline_vs_trained.md`](results/final_baseline_vs_trained.md), [`results/ablation_hack_penalty.json`](results/ablation_hack_penalty.json)
5. Interactive judge checklist: app `/` and `/history`

Deterministic run order:
```bash
python evaluate_models.py --offline --tasks easy,medium,hard,frontier --episodes 3 --seeds 41,42,43 --output results/final_baseline_vs_trained.md --trained-model offline_stub
python ablation_eval.py
python -m server.app   # then GET /export/scorecard
```

---

## Evidence Snapshot

### Proof of learning (primary training run)

| Task | Base Reward | Trained Reward | Gain |
| :--- | ---: | ---: | ---: |
| Easy | 0.42 | 0.65 | +56% |
| Medium | 0.02 | 0.53 | +1,872% |
| Hard | -0.21 | 0.39 | +284% |

Source: [`results/baseline_vs_trained_colab.json`](results/baseline_vs_trained_colab.json)

### Deterministic multi-seed reproducibility pack

| Task | Baseline Mean ± Std | Trained Mean ± Std | Delta |
| :--- | ---: | ---: | ---: |
| Easy | 0.314 ± 0.089 | 0.981 ± 0.055 | +212.0% |
| Medium | -0.954 ± 0.293 | 0.604 ± 0.246 | +163.3% |
| Hard | -0.717 ± 0.260 | -0.133 ± 0.327 | +81.4% |
| Frontier | -0.616 ± 0.070 | -0.131 ± 0.058 | +78.8% |

Source: [`results/final_baseline_vs_trained.md`](results/final_baseline_vs_trained.md)

### Frontier interpretation (important)

Frontier is intentionally the hardest regime: high-risk governance gates, multilingual/noisy inputs, tool failures, and delayed consequences.  
In this mode, we prioritize **safety-constrained improvement** over absolute score inflation.

- Reward still improves versus baseline (`+78.8%` in deterministic reruns).
- Frontier success can remain low while behavior quality improves, because blocked unsafe autonomy and safe handoffs are explicitly rewarded.
- We treat frontier as a stress-test track for safe autonomy, not a “quick-win” metric track.

### Anti-reward-hacking evidence

Removing the anti-spam penalty increases spam-policy reward by `+0.92`, confirming the safeguard closes a real loophole.

Source: [`results/ablation_hack_penalty.json`](results/ablation_hack_penalty.json)

![Reward Curves](results/reward_curves.png)
*Reward rises across curriculum phases instead of relying on prompt-only behavior.*

---

## What This Tests

- **Environment benchmark, not a wrapper:** state includes SLA drift, queue pressure, sentiment, governance, and tool health.
- **Partially observable world model:** agents act on incomplete, noisy, multilingual inputs with delayed consequences.
- **Policy-aware autonomy:** high-risk cases require evidence-gated actions (`tool_call`, `human_review_required`, `legal_hold`).
- **Failure recovery:** provider-style latency/timeout/rate-limit/fallback dynamics are simulated and scored.
- **Measurable robustness:** reward gains, reproducible runs, and anti-hacking ablations are all included.

---

## Architecture at a Glance

### Environment core
- [`src/customer_support_env.py`](src/customer_support_env.py): OpenEnv environment, task progression, reward shaping, telemetry
- [`openenv.yaml`](openenv.yaml): task declarations and benchmark manifest

### World-model realism layers
- [`src/mock_data_fixtures.py`](src/mock_data_fixtures.py): multi-domain scenarios (`ecommerce`, `telecom`, `healthcare_insurance`, `travel`)
- [`src/voice_stack.py`](src/voice_stack.py): noisy/multilingual text normalization abstraction
- [`src/policy_rules.py`](src/policy_rules.py): high-risk policy detection and gating

### Tool and reliability stack
- [`src/toolhub.py`](src/toolhub.py): evidence APIs (`policy_lookup`, `fraud_screen`, `kyc_verify`, etc.)
- [`src/mock_api_stack.py`](src/mock_api_stack.py): degraded provider behavior (timeouts, rate-limits, fallback)

### Evaluation and scorecards
- [`train.py`](train.py), [`frontierops_training_lab.ipynb`](frontierops_training_lab.ipynb): GRPO training pipeline
- [`evaluate_models.py`](evaluate_models.py): baseline vs trained evaluation
- [`ablation_eval.py`](ablation_eval.py): reward-hacking and governance ablation
- [`server/app.py`](server/app.py): `/scorecard`, `/export/scorecard`, `/providers/health`

---

## Benchmark Scenario Coverage

### Task modes (7)
`easy`, `medium`, `hard`, `chaos`, `multi_agent_triage`, `multi_agent_resolver`, `frontier`

### High-risk categories (6)
`pii_exposure`, `fraud_risk`, `account_takeover`, `prompt_injection`, `legal_threat`, `medical_safety`

### First-class safety actions
`tool_call`, `human_review_required`, `legal_hold`, `resolve`, `escalate`

### Telemetry signals
`safe_handoff`, `unsafe_action_blocked`, `wrongful_autonomy`, `governance_blocks`, `tool_calls`, `tool_fallbacks`

---

## Hackathon Theme Alignment

| Theme | What is implemented | Where to verify |
| :--- | :--- | :--- |
| #1 Multi-Agent Interactions | Role-specialized triage/resolver with incentive-modeled coordination and anti-dumping penalties | `src/customer_support_env.py` (`coordination_routing_bonus`, `coalition_safety_bonus`, `missed_handoff_penalty`, `escalation_dumping_penalty`, `handoff_contradiction_penalty`) |
| #3.1 World Modeling | Partially observable enterprise state + tool calls + degraded providers + governance gates | `src/customer_support_env.py`, `src/toolhub.py`, `src/mock_api_stack.py`, `src/policy_rules.py` |
| #4 Self-Improvement (roadmap) | Curriculum progression and planned self-generated hard-case expansion | `train.py`, roadmap section below |

### Theme #1: Multi-Agent Interactions
- Specialized roles in `multi_agent_triage` and `multi_agent_resolver`
- Incentive-modeled coordination via explicit reward terms:
  - `coordination_routing_bonus`, `coalition_safety_bonus`
  - `missed_handoff_penalty`, `escalation_dumping_penalty`, `handoff_contradiction_penalty`
- Coordination behavior under queue pressure, SLA pressure, and risk-handoff trade-offs

### Theme #3.1: World Modeling (Professional Tasks)
- Partially observable enterprise operations world
- Tool/API orchestration with real failure dynamics
- Evidence-gated decision policy in high-stakes states

### Theme #4: Self-Improvement (Roadmap)
- Adaptive curriculum from easy to frontier
- Planned self-generated edge-case tasks for recursive capability growth

---

## Reproducibility and Judge Verification

### Local quick start
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
openenv validate .
python test_env.py
python -m server.app
```

### Repro pipeline
```bash
python train.py --model Qwen/Qwen2.5-0.5B-Instruct
python evaluate_models.py --offline --tasks easy,medium,hard,frontier --episodes 3 --seeds 41,42,43 --output results/final_baseline_vs_trained.md
python ablation_eval.py
```

### Artifact index
- Metrics: [`results/baseline_vs_trained_colab.json`](results/baseline_vs_trained_colab.json)
- Repro table: [`results/final_baseline_vs_trained.md`](results/final_baseline_vs_trained.md)
- Curves: [`results/reward_curves.png`](results/reward_curves.png)
- Summary stats: [`results/final_summary_stats.json`](results/final_summary_stats.json)
- Ablations: [`results/ablation_hack_penalty.json`](results/ablation_hack_penalty.json)
- Judge one-pager: [`results/judge_scorecard.md`](results/judge_scorecard.md)
- Judge Q&A: [`JUDGE_QA.md`](JUDGE_QA.md)

---

## Differentiation vs Common Baselines

- **Not a chatbot wrapper:** policy is trained in an interactive environment, not prompt-tuned on static responses.
- **Not simple RAG:** success depends on action sequencing, escalation judgment, and tool-grounded evidence.
- **Not a toy environment:** includes adversarial and high-risk governance states plus degraded APIs.
- **Not a static benchmark:** dynamic queue/SLA trajectories produce non-trivial long-horizon credit assignment.
- **Not prompt engineering only:** measurable pre/post training improvements and ablation-backed reward design.

---

## Roadmap to Frontier Scale

1. Upgrade policy models to 7B-8B class for deeper long-horizon reasoning.
2. Expand scenario distribution with larger multilingual, adversarial, and domain-specific episodes.
3. Add self-play style ticket generation and adaptive difficulty for Theme #4.
4. Add stronger multi-agent negotiation and coalition tasks to intensify Theme #1.

---

## Submission Compliance

- Uses OpenEnv latest-compliant environment API and manifest
- Includes GRPO/TRL training script and reproducible evaluation pipeline
- Ships training evidence (reward curves + before/after results)
- Hosted and runnable via Hugging Face Space
- Provides concise judge artifacts and deterministic rerun path

---

*FrontierOps Arena · OpenEnv Hackathon Submission · Punith S*
