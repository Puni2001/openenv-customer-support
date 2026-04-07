# Customer Support Ticket Resolution Environment

An OpenEnv-compliant environment for training AI agents to handle customer support tickets.

## Environment Description

This environment simulates a customer support queue where an AI agent must:
- Categorize tickets correctly
- Prioritize based on sentiment and SLA constraints
- Resolve issues using a knowledge base
- Escalate appropriately when needed

## Tasks

### Easy: Ticket Categorization
Classify tickets into correct categories (technical, billing, account, feature_request, complaint)

### Medium: Categorization + Prioritization
Categorize tickets AND set appropriate priority based on SLA deadlines and customer sentiment

### Hard: Full Resolution
Resolve tickets using knowledge base, manage sentiment recovery, and make escalation decisions

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN=your_token_here
export MODEL_NAME=gpt-3.5-turbo

# Run evaluation
python inference.py
