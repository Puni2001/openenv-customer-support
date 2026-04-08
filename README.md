---
title: OpenEnv Customer Support
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Customer Support Ticket Resolution Environment

An OpenEnv-compliant environment for training AI agents to handle real-world customer support workflows. This environment models a multi-step support process including categorization, prioritization, resolution, and escalation.

## Real-World Utility

Customer support is a high-volume, critical business function. Automating ticket triage and resolution can significantly reduce SLA breaches and improve customer satisfaction. This environment provides a realistic simulation of:
- **Triage**: Correctly identifying technical vs. billing issues.
- **SLA Management**: Prioritizing urgent cases based on customer sentiment and account status.
- **Knowledge Base Retrieval**: Using simulated guidelines to resolve common issues.
- **Escalation Logic**: Identifying when a human or manager needs to intervene.

## Environment Specification

### Observation Space (Pydantic Model)
The agent receives an `Observation` object containing:
- `current_ticket`: A `Ticket` object with `description`, `category`, `priority`, and `sentiment`.
- `tickets_remaining`: Number of tickets left in the queue.
- `tickets_handled`: Number of tickets successfully processed.
- `current_sla_status`: "ok", "warning", or "breached".
- `recent_actions`: History of actions taken in the current episode.

### Action Space (Pydantic Model)
The agent must choose an `Action`:
- `action_type`: One of `categorize`, `prioritize`, `resolve`, `escalate`, `request_info`.
- `value`: The target value (e.g., the category name or resolution notes).
- `reasoning`: A brief explanation of the decision.

## Tasks

| Task | Level | Description | Success Criteria |
| :--- | :--- | :--- | :--- |
| **Easy** | Triage | Correctly categorize incoming tickets. | 100% accurate classification. |
| **Medium** | Priority | Categorize + Set priority based on SLA/Sentiment. | Accurate classification and priority meeting SLA. |
| **Hard** | Resolution | Categorize + Prioritize + Resolve or Escalate. | Tickets resolved using KB or escalated appropriately. |

## Compliance & Validation

This environment follows the OpenEnv specification:
- [x] Valid `openenv.yaml`
- [x] Typed Pydantic models for Action/Observation
- [x] Implements `reset()`, `step()`, and `state()`
- [x] Mandatory `inference.py` stdout format
- [x] Containerized with Dockerfile

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN=your_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# Run evaluation
python inference.py
```

## Validation

To run the OpenEnv validator:
```bash
openenv validate .
```
