---
title: OpenEnv | AI Support Envoy
emoji: 🎧
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
---

# 🎧 AI Support Envoy

> An OpenEnv-compliant RL environment for training AI agents on enterprise customer support workflows.

**Hackathon Themes:** World Modeling — Professional Tasks (#3.1) + Multi-Agent Interactions (#1)

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

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://huggingface.co/spaces/punith2001/ai-support-envoy
cd ai-support-envoy
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Add your HF token to .env
echo 'HF_TOKEN=hf_your_token_here' >> .env
echo 'BASE_MODEL_NAME=Qwen/Qwen2.5-72B-Instruct' >> .env
echo 'TRAINED_MODEL_NAME=punith2001/openenv-customer-support-model' >> .env

# 3. Validate the environment
openenv validate .

# 4. Run inference baseline
python inference.py

# 5. Start the server + UI
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
# Full curriculum: easy → medium → hard
python train.py

# Single task level
python train.py --task hard --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3

# Push trained model to HF Hub
python train.py --push-to-hub --hub-repo your-username/ai-support-envoy-model
```

See `train_colab.ipynb` for the full Colab notebook with reward curves and measured baseline-vs-trained outputs.

### Reproducible Evaluation (Baseline vs Trained)

```bash
python evaluate_models.py \
  --base-model Qwen/Qwen2.5-72B-Instruct \
  --trained-model punith2001/openenv-customer-support-model \
  --tasks easy,medium,hard \
  --seeds 41,42,43,44,45 \
  --output results/baseline_vs_trained.json
```

This writes machine-readable metrics to `results/baseline_vs_trained.json` (no placeholders).

---

## Server API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Interactive dashboard UI |
| POST | `/reset` | Start new episode (`task_level`, `seed`) |
| POST | `/step` | Execute action |
| GET | `/state` | Current state + reward history |
| GET | `/metrics` | Aggregated metrics across all sessions |
| GET | `/history` | **Run History Dashboard** to view uploaded reward curves and past runs |
| POST | `/api/runs` | API to save a run with logs and timestamp |
| GET | `/docs` | Auto-generated API docs (Swagger) |

Sessions are isolated via `X-Session-Id` header.

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

---

*AI Support Envoy · Meta PyTorch OpenEnv Hackathon · Punith S*
