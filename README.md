---
title: OpenEnv | AI Support Envoy
emoji: 🎧
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
---

# 🎧 AI Support Envoy

> **📖 Technical Blog & Report:** [Read our full Grand Finale Submission Report on Hugging Face](https://huggingface.co/spaces/punith2001/openenv-customer-support/discussions/1)

[![Watch the Demo](https://img.youtube.com/vi/BQKfDsODfFk/0.jpg)](https://youtu.be/BQKfDsODfFk)
*Click above to watch the AI Support Envoy in action.*

**An OpenEnv-compliant RL environment for training AI agents on enterprise customer support workflows.**

**Hackathon Themes:** World Modeling — Professional Tasks (#3.1) + Multi-Agent Interactions (#1)

---

## Final Performance Report (Verified Proof)

**Judge fast path:**
- One-page evidence summary: `results/judge_scorecard.md`
- Judge Q&A pack: `JUDGE_QA.md`
- Full interactive checklist (same facts as below): open the app UI → **Judge Checklist** section on the dashboard (`/` → scroll to *Judge — Complete Feature Checklist*), or **Run History** (`/history`) → *Judge — Repo Artifacts & Scripts*.

## Judge: Full Feature Index (manual review)

Use this section so nothing in the codebase is “hidden” from reviewers.

### Task modes (7)

`easy`, `medium`, `hard`, `chaos`, `multi_agent_triage`, `multi_agent_resolver`, `frontier` — all declared in [`openenv.yaml`](openenv.yaml) and implemented in [`src/customer_support_env.py`](src/customer_support_env.py).

### Six high-risk categories

Detected from ticket text; governance gate in [`src/policy_rules.py`](src/policy_rules.py):

| Category | Purpose |
| :--- | :--- |
| `pii_exposure` | Sensitive data handling; blocks naive resolve without evidence |
| `fraud_risk` | Requires fraud tool evidence before autonomy |
| `account_takeover` | Requires KYC-style evidence |
| `prompt_injection` | Adversarial instruction patterns |
| `legal_threat` | Triggers `legal_hold` path |
| `medical_safety` | High-stakes health wording |

First-class actions: `tool_call`, `human_review_required`, `legal_hold` (plus `resolve`, `escalate`, …). Telemetry in env state: `safe_handoff`, `unsafe_action_blocked`, `wrongful_autonomy`, `governance_blocks`, `tool_calls`, `tool_fallbacks`.

### Four industry domain packs

Defined in [`src/mock_data_fixtures.py`](src/mock_data_fixtures.py): **ecommerce**, **telecom**, **healthcare_insurance**, **travel** — intents, policy snippets, multilingual / adversarial customer lines.

### Mock `tool_call` APIs (evidence before autonomy)

Implemented via [`src/toolhub.py`](src/toolhub.py) + provider-style layer [`src/mock_api_stack.py`](src/mock_api_stack.py) (latency, rate-limit, timeout, fallback). **Tool names:** `policy_lookup`, `fraud_screen`, `kyc_verify`, `trust_safety_review`, `legal_escalation`, `customer_history`, `payment_lookup`, `order_lookup`. Optional richer customer fields use **Faker** when installed (`requirements.txt`).

### Voice / multilingual (mock)

[`src/voice_stack.py`](src/voice_stack.py): mock ASR/TTS, text normalization, light language hints for code-mixed input.

### RL, anti-hacking, ablations

- Training: [`train.py`](train.py) (GRPO / TRL), curriculum including `frontier`; notebook [`train_colab.ipynb`](train_colab.ipynb).
- Anti–reward-hacking: urgent-spam penalty, fake-empathy guard (in env reward).
- Ablations: [`ablation_eval.py`](ablation_eval.py) → [`results/ablation_hack_penalty.json`](results/ablation_hack_penalty.json) (penalty on/off + governance ablation summary).

### Evaluation & reproducibility

[`evaluate_models.py`](evaluate_models.py): offline multi-seed runs; outputs [`results/final_baseline_vs_trained.md`](results/final_baseline_vs_trained.md), [`results/final_baseline_vs_trained.json`](results/final_baseline_vs_trained.json), [`results/final_summary_stats.json`](results/final_summary_stats.json); markdown includes **Safety and Governance Scorecard** table.

### Scorecards & server API

[`server/app.py`](server/app.py): `GET /scorecard`, `GET /export/scorecard` (writes [`results/scorecard_report.json`](results/scorecard_report.json)), `GET /providers/health` (per session). Rollups in [`src/telemetry.py`](src/telemetry.py).

### UI for judges

- [`server/dashboard.html`](server/dashboard.html): all 7 levels, Frontier stack, **Judge Checklist** block, reward table incl. governance rows, API list incl. scorecard endpoints.
- [`server/history.html`](server/history.html): scorecard export, link to dashboard checklist, **Judge — Repo Artifacts & Scripts** list.

### Primary vs fallback evidence (framing)

- **Primary stronger run:** [`results/baseline_vs_trained_colab.json`](results/baseline_vs_trained_colab.json) + [`results/reward_curves.png`](results/reward_curves.png).
- **Deterministic reruns:** `results/final_*`, ablation JSON, scorecard export, judge one-pager + Q&A as above.

After a full curriculum training run (Easy → Medium → Hard) using **GRPO**, the AI Support Envoy achieved massive gains over the base model.
All final reproducible runs in this repository use **Qwen/Qwen2.5-0.5B-Instruct** as the practical baseline for solo compute.

| Task Level | Base Reward (Avg) | Trained Reward (Avg) | Improvement |
| :--- | :---: | :---: | :---: |
| **Easy** | 0.42 | 0.65 | +56% |
| **Medium** | 0.02 | 0.53 | **+1,872%** |
| **Hard** | -0.21 | 0.39 | **+284%** |

### Training Mastery (Reward Curves)
![Reward Curves](results/reward_curves.png)
*Figure 1: Training progress across the 3 curriculum phases. Note the agent successfully climbing from negative rewards to consistent mastery.*

### Raw Evidence (results/baseline_vs_trained_colab.json)
```json
{
  "before": { "easy": 0.420, "medium": 0.027, "hard": -0.215 },
  "after":  { "easy": 0.655, "medium": 0.532, "hard": 0.396 }
}
```

### Repro Pack (Solo Budget Run)
The repository includes a deterministic evaluation/ablation pack for constrained-compute reruns.
Use this as a reproducibility fallback; primary judging evidence should come from your strongest real training run.

| Task Level | Baseline Mean ± Std | Trained Mean ± Std | Delta |
| :--- | :---: | :---: | :---: |
| **Easy** | 0.314 ± 0.089 | 0.981 ± 0.055 | **+212.0%** |
| **Medium** | -0.954 ± 0.293 | 0.604 ± 0.246 | **+163.3%** |
| **Hard** | -0.717 ± 0.260 | -0.133 ± 0.327 | **+81.4%** |
| **Frontier** | -0.616 ± 0.070 | -0.131 ± 0.058 | **+78.8%** |

Anti-hacking ablation (`results/ablation_hack_penalty.json`) shows that removing the over-prioritization penalty increases reward for the spam policy by `+0.92`, confirming that the penalty closes a real reward-hacking loophole.

---

## Future Roadmap (With Additional Credits)
With more compute, we are ready to scale this project further:
1. **Model Scaling**: Upgrade from Qwen-0.5B to **Llama-3.1-8B** or **Qwen-2.5-7B** for deeper reasoning.
2. **Dataset Expansion**: Increase training samples from 200 to **2,000 per task** for smoother convergence.
3. **Recursive Self-Improvement**: Implement Theme #4 by allowing the agent to generate its own "Edge Case" tickets to train on.

---


---

## What is this?

A training gym for AI agents. The agent plays the role of a customer support specialist at a company. Customers send in tickets — complaints, billing issues, technical problems. The agent must triage, prioritize, and resolve them correctly under real constraints like SLA deadlines, VIP customers, and negative sentiment.

The environment scores every action with a dense reward signal, so the agent gets feedback at every step — not just at the end. This makes it ideal for GRPO/PPO training.

---

## Task Levels

| Level | What the agent does | Key reward signals |
|---|---|---|
| `easy` | Classify ticket into correct category | +0.40 correct, -0.30 wrong |
| `medium` | Classify + set priority with SLA awareness | Priority × SLA urgency multiplier |
| `hard` | Full resolution using knowledge base | KB quality + empathy + SLA compliance |
| `chaos` | Handle ticket storm (8 tickets, dynamic SLAs) | Urgency-weighted, VIP-boosted |
| `multi_agent_triage` | Triage specialist — categorize and route | Routing accuracy |
| `multi_agent_resolver` | Resolver specialist — resolve pre-triaged tickets | Resolution quality |
| `frontier` | Multi-domain, multilingual, high-risk evidence-gated support | Governance-safe autonomy + handoff correctness |

---

## Reward Model (Dense + Shaped)

```
categorize:   +0.40 correct | +0.10 valid-but-wrong | -0.30 invalid
prioritize:   +0.35 × sla_urgency | +0.10 off-by-one | -0.25 wrong
              -0.30 extra penalty for under-prioritizing urgent tickets
resolve:      +0.50 × kb_quality | +0.15 empathy bonus | +0.10 sla_ok
              -0.20 sla_breach | ±0.10 repeat contact handling
escalate:     +0.35 correct | +0.05 reasoning bonus | -0.40 unnecessary
step_cost:    -0.01 per step (efficiency incentive)

SLA urgency multiplier: 1.5× breached | 1.2× warning | 1.0× ok
VIP customers: priority escalated one level automatically
```

### Frontier Governance Extensions
- Voice/text ingestion abstraction with code-mixed language detection (`src/voice_stack.py`)
- Multi-industry fixtures for ecommerce, telecom, healthcare/insurance, and travel (`src/mock_data_fixtures.py`)
- Six high-risk category detection + evidence-gated policy decisions (`src/policy_rules.py`)
- Mock tool hub for order/payment/policy/fraud/history/legal/trust-safety workflows (`src/toolhub.py`)
- Provider-style mock APIs with latency/failure/fallback simulation (`src/mock_api_stack.py`)
- Telemetry metrics in env state: `safe_handoff`, `unsafe_action_blocked`, `wrongful_autonomy`, `governance_blocks`
- Evidence-first tool flow with `tool_call` actions (`policy_lookup`, `fraud_screen`, `kyc_verify`, `trust_safety_review`) before final decisions
- Reliability telemetry for tools: `tool_calls`, `tool_fallbacks`, and scorecard `tool_fallback_rate`

---

## Quick Start (Judge Run Order)

```bash
# 1. Clone and set up
git clone https://huggingface.co/spaces/punith2001/ai-support-envoy
cd ai-support-envoy
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Add your HF token to .env
echo 'HF_TOKEN=hf_your_token_here' >> .env
echo 'BASE_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct' >> .env
echo 'TRAINED_MODEL_NAME=punith2001/openenv-customer-support-model' >> .env

# 3. Validate OpenEnv manifest + interfaces
openenv validate .

# 4. Smoke tests (environment + API)
python test_env.py
python -m server.app &
python test_api.py

# 5. Baseline inference run (strict [START]/[STEP]/[END] logging)
python inference.py

# 6. Start the server + UI
python -m server.app
# Open http://localhost:7860
```

---

## Environment API

```python
from src.customer_support_env import CustomerSupportEnv, Action

# Create environment
env = CustomerSupportEnv(task_level="hard", seed=42)
obs = env.reset()

# Current ticket
print(obs.current_ticket.description)   # "App crashes when uploading files"
print(obs.current_ticket.sentiment)     # -0.72
print(obs.current_sla_status)           # "warning"

# Take an action
action = Action(
    action_type="resolve",
    value="Apologize for the frustration. Diagnosed error logs, restarted service, cleared cache.",
    reasoning="Negative sentiment + technical issue — KB-aligned with empathy"
)
obs, reward, done, info = env.step(action)

print(reward)                    # 0.74
print(info["reward_breakdown"])  # {'resolution_quality': 0.5, 'empathy_bonus': 0.15, ...}

# Curriculum learning
env.record_episode_score(0.85)
print(env.suggest_next_level())  # "chaos"
```

---

## Training (GRPO with TRL)

```bash
# Full curriculum: easy → medium → hard (practical solo default model)
python train.py --model Qwen/Qwen2.5-0.5B-Instruct

# Single task level
python train.py --task hard --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3

# Push trained model to HF Hub
python train.py --push-to-hub --hub-repo your-username/ai-support-envoy-model
```

### Fast Reproducibility Pipeline (Local/Offline)

```bash
# 1) Sanity run
python train.py --task easy --curriculum easy --model sshleifer/tiny-gpt2 --epochs 1 --samples 8 --batch-size 1

# 2) Compact curriculum run
python train.py --model sshleifer/tiny-gpt2 --curriculum easy,medium,hard --epochs 1 --samples 12 --batch-size 1

# 3) 3-seed reproducible evaluation
python evaluate_models.py --offline --tasks easy,medium,hard,frontier --episodes 3 --seeds 41,42,43 --output results/final_baseline_vs_trained.md

# 4) Anti-hacking ablation
python ablation_eval.py
```

See `train_colab.ipynb` for the full Colab notebook with reward curves and measured baseline-vs-trained outputs.

### Reproducible Evaluation (Baseline vs Trained)

```bash
python evaluate_models.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --trained-model punith2001/openenv-customer-support-model \
  --tasks easy,medium,hard \
  --seeds 41,42,43,44,45 \
  --output results/final_baseline_vs_trained.md
```

This writes:
- Markdown table: `results/final_baseline_vs_trained.md`
- Machine-readable JSON: `results/final_baseline_vs_trained.json`

### Training Artifacts
*   **Metrics**: [results/baseline_vs_trained_colab.json](results/baseline_vs_trained_colab.json)
*   **Visuals**: [results/reward_curves.png](results/reward_curves.png)
*   **Summary stats**: [results/final_summary_stats.json](results/final_summary_stats.json)
*   **Ablation**: [results/ablation_hack_penalty.json](results/ablation_hack_penalty.json)
*   **Comparison plot**: [results/final_reward_comparison.png](results/final_reward_comparison.png)
*   **Judge one-pager**: [results/judge_scorecard.md](results/judge_scorecard.md)
*   **Judge Q&A**: [JUDGE_QA.md](JUDGE_QA.md)

### Notes on Evidence Tracks
- `results/baseline_vs_trained_colab.json` + `results/reward_curves.png`: prior full run evidence
- `results/final_*` + ablation files: latest deterministic reproducibility pack for judge reruns
- `results/scorecard_report.json`: exported production-style SLO/safety/business KPI scorecard

---

## Server API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Interactive dashboard UI |
| POST | `/reset` | Start new episode (`task_level`, `seed`) |
| POST | `/step` | Execute action |
| GET | `/state` | Current state + reward history |
| GET | `/metrics` | Aggregated metrics across all sessions |
| GET | `/scorecard` | SLO + safety + business KPI scorecard |
| GET | `/export/scorecard` | Persist scorecard to `results/scorecard_report.json` |
| GET | `/providers/health` | Provider health status (requires `X-Session-Id`) |
| GET | `/history` | **Run History Dashboard** to view uploaded reward curves and past runs |
| POST | `/api/runs` | API to save a run with logs and timestamp |
| GET | `/docs` | Auto-generated API docs (Swagger) |

Sessions are isolated via `X-Session-Id` header.

### Web UI Coverage
- `server/dashboard.html` now exposes all 7 task levels including `frontier`, plus Frontier Stack, governance-aware reward rows, and reliability endpoint visibility.
- `server/history.html` now includes quick scorecard export actions and judge-review helper cues.

```bash
# Example: run a full episode via curl
SESSION=$(curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_level":"hard"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: $SESSION" \
  -d '{"action_type":"resolve","value":"Apologize and restart the service","reasoning":"technical issue"}'
```

---

## Project Structure

```
ai-support-envoy/
├── src/
│   └── customer_support_env.py   # Core environment (6 task levels, reward model, curriculum)
├── tasks/
│   └── grader.py                 # Programmatic graders for all levels
├── server/
│   ├── app.py                    # FastAPI server
│   ├── dashboard.html            # Interactive UI
│   ├── history.html              # Run History dashboard
│   └── static/                   # Static directory for saved runs & images
├── inference.py                  # LLM agent + baseline runner
├── train.py                      # GRPO training script (TRL)
├── evaluate_models.py            # Reproducible baseline-vs-trained evaluation
├── train_colab.ipynb             # Colab notebook with reward curves
├── openenv.yaml                  # OpenEnv config
├── Dockerfile                    # HF Space deployment
└── .env                          # HF_TOKEN, MODEL_NAME, API_BASE_URL
```

---

## Hackathon Compliance

- ✅ `openenv validate .` passes
- ✅ HuggingFace Router only (`https://router.huggingface.co/v1`)
- ✅ OpenAI client for all LLM calls
- ✅ Strict `[START]` / `[STEP]` / `[END]` stdout format
- ✅ Training script with GRPO via TRL (`train.py`)
- ✅ Colab notebook with reward curves (`train_colab.ipynb`)
- ✅ Multi-seed reproducibility pack (`results/final_baseline_vs_trained.md`, `results/final_summary_stats.json`)
- ✅ Anti-hacking ablation artifact (`results/ablation_hack_penalty.json`)

---

*AI Support Envoy · Meta PyTorch OpenEnv Hackathon · Punith S*
