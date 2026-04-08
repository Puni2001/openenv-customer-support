---
title: OpenEnv | AI Support Envoy
emoji: 🎧
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
---

# 🏢 AI Support Envoy: Enterprise Customer Support Environment

An OpenEnv-compliant environment designed to train agents for high-stakes, multi-step customer support triage and resolution. This environment bridges the gap between raw LLM chat and production-ready support agents.

![Landing Page](/Landing.png) 
> *The professional dashboard included in the Space provides real-time health and API status.*

## 🌟 Real-World Utility
Customer support is a critical bottleneck for growing enterprises. This environment provides a verifiable training ground for:
- **Intelligent Triage**: Automatically categorizing technical vs. billing issues with high precision.
- **SLA Enforcement**: Dynamic prioritization based on customer sentiment and urgency.
- **Complexity Management**: Handling escalations to specialized teams (DevOps, Billing, Compliance) when standard resolution fails.
- **Inference Resilience**: Built with production-grade logic that handles API rate limits and transient network errors gracefully.

## 📊 Benchmark results (Gemini 1.5 Flash)
Our baseline agent achieved verified success across multiple difficulty levels:
- **Easy (Triage)**: 100% Accuracy (Score: 1.00)
- **Medium (Priority)**: Success with partial rewards
- **Hard (Resolution)**: Success (Score: 0.27) with complex escalation handling.

## 🛠️ Environment Specification

### Observation Space
The agent receives a rich state context including:
- `current_ticket`: Full ticket details (description, category, priority, sentiment).
- `tickets_remaining`: Current queue depth.
- `current_sla_status`: ok | warning | breached.
- `recent_actions`: History of the current episode (prevents loops).

### Action Space
Agents interact via a structured JSON bridge:
- `action_type`: `categorize` | `prioritize` | `resolve` | `escalate`
- `value`: The decision payload (e.g., category name or resolution steps).
- `reasoning`: Chain-of-thought justification for the action.

## 🏆 Hackathon Compliance
- ✅ **Hugging Face Router Compatible**: No external API infrastructure needed.
- ✅ **Strict Stdout Format**: Implements the mandatory `[START]`, `[STEP]`, `[END]` logging.
- ✅ **Verifiable Rewards**: Programmatic grading (0.0 to 1.0 scale).
- ✅ **Validated**: Passed `openenv validate .` verification.

## 🚀 Quick Start
```bash
# Verify the environment locally
./venv/bin/openenv validate .

# Run the inference baseline
./venv/bin/python3 inference.py
```

---
*Created for the OpenEnv Hackathon | Submission by punith2001*
