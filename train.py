#!/usr/bin/env python3
"""
AI Support Envoy — GRPO Training Script
=========================================
Uses TRL's GRPOTrainer to fine-tune an LLM on the customer support environment.
The environment reward signal drives policy improvement.

Usage:
  python train.py                          # default: Qwen2.5-0.5B, easy→hard curriculum
  python train.py --task hard              # single task
  python train.py --model Qwen/Qwen2.5-1.5B-Instruct

Colab: See train_colab.ipynb for the full notebook with reward curves.
"""

import argparse
import json
import os
import random
from datetime import datetime, timedelta
from typing import List, Dict

import numpy as np
import torch

# ── Reward function (environment-grounded) ───────────────────

def make_reward_fn(task_level: str):
    """
    Returns a reward function compatible with TRL's GRPOTrainer.
    Each completion is scored by the environment's grader.
    """
    from src.customer_support_env import CustomerSupportEnv, Action, Ticket, TicketCategory, Priority

    def reward_fn(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
        rewards = []
        ticket_payloads = kwargs.get("ticket_payload", [])
        for idx, completion in enumerate(completions):
            try:
                # Parse LLM output as action JSON
                text = completion.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                start, end = text.find("{"), text.rfind("}") + 1
                data = json.loads(text[start:end]) if start >= 0 else {}

                action = Action(
                    action_type=data.get("action_type", "request_info"),
                    value=str(data.get("value", "")),
                    reasoning=str(data.get("reasoning", ""))
                )

                # Score against the same ticket context used in the prompt.
                payload = ticket_payloads[idx] if idx < len(ticket_payloads) else {}
                if isinstance(payload, str):
                    payload = json.loads(payload)
                ticket = Ticket(
                    customer_id=payload.get("customer_id", "train_customer"),
                    category=TicketCategory(payload.get("category", "technical")),
                    description=payload.get("description", "Training sample ticket"),
                    sentiment=float(payload.get("sentiment", 0.0)),
                    priority=Priority(payload.get("priority", "medium")),
                    created_at=datetime.now(),
                    sla_deadline=datetime.now() + timedelta(hours=max(1, int(payload.get("sla_hours", 4)))),
                    previous_contacts=int(payload.get("previous_contacts", 0)),
                    is_vip=bool(payload.get("is_vip", False)),
                )

                env = CustomerSupportEnv(task_level=task_level)
                reward, _ = env._calculate_reward(action, ticket)
                rewards.append(float(reward))

            except Exception:
                rewards.append(-0.5)  # penalty for malformed output

        return rewards

    return reward_fn


# ── Dataset generation ───────────────────────────────────────

def generate_dataset(task_level: str, n_samples: int = 500) -> List[Dict]:
    """Generate training prompts from the environment."""
    from src.customer_support_env import CustomerSupportEnv, KNOWLEDGE_BASE

    env = CustomerSupportEnv(task_level=task_level)
    samples = []

    SYSTEM_PROMPTS = {
        "easy": "You are a customer support triage agent. Classify the ticket into exactly one category: technical, billing, account, feature_request, or complaint. Respond in JSON: {\"action_type\": \"categorize\", \"value\": \"<category>\", \"reasoning\": \"<why>\"}",
        "medium": "You are a support ops agent. Set the correct priority: low, medium, high, or urgent. Respond in JSON: {\"action_type\": \"prioritize\", \"value\": \"<priority>\", \"reasoning\": \"<why>\"}",
        "hard": "You are a support resolution specialist. Resolve the ticket using KB steps or escalate (if sentiment < -0.7 or category is complaint). Respond in JSON: {\"action_type\": \"resolve|escalate\", \"value\": \"<resolution>\", \"reasoning\": \"<why>\"}",
        "chaos": "You are handling a ticket storm. Resolve or escalate efficiently. Respond in JSON: {\"action_type\": \"resolve|escalate\", \"value\": \"<resolution>\", \"reasoning\": \"<why>\"}",
        "frontier": "You are a governance-aware support agent. Use tool_call to collect evidence (fraud_screen, kyc_verify, policy_lookup, trust_safety_review) before final actions. Choose tool_call/resolve/escalate/human_review_required/legal_hold based on risk and evidence. Respond in JSON: {\"action_type\": \"tool_call|resolve|escalate|human_review_required|legal_hold\", \"value\": \"<tool or decision>\", \"reasoning\": \"<evidence and policy>\"}",
    }

    system = SYSTEM_PROMPTS.get(task_level, SYSTEM_PROMPTS["hard"])

    for _ in range(n_samples):
        obs = env.reset()
        ticket = obs.current_ticket
        if not ticket:
            continue

        # Do not leak ground-truth labels like Category/Priority in the prompt.
        # Also avoid implicit leakage via KB steps for triage tasks.
        meta = []
        meta.append(f"Ticket: {ticket.description}")
        meta.append(f"Sentiment: {ticket.sentiment:.2f}")
        meta.append(f"SLA Status: {obs.current_sla_status}")
        meta.append(f"VIP: {ticket.is_vip}")
        meta.append(f"Previous Contacts: {ticket.previous_contacts}")
        
        if task_level in ("hard", "chaos", "frontier"):
            kb = KNOWLEDGE_BASE.get(ticket.category.value, {})
            steps = ", ".join(kb.get("steps", []))
            meta.append(f"KB Steps: {steps}")
            meta.append(f"High Risk Flags: {', '.join(ticket.high_risk_flags) if ticket.high_risk_flags else 'none'}")
            meta.append(f"Governance Hint: {env._build_observation(ticket).governance_hint}")
            
        user = "\n".join(meta)

        samples.append({
            "prompt": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "task_level": task_level,
            "ticket_payload": json.dumps({
                "customer_id": ticket.customer_id,
                "category": ticket.category.value,
                "description": ticket.description,
                "sentiment": ticket.sentiment,
                "priority": ticket.priority.value,
                "sla_hours": max(1, int((ticket.sla_deadline - ticket.created_at).total_seconds() / 3600)),
                "is_vip": ticket.is_vip,
                "previous_contacts": ticket.previous_contacts,
            })
        })

    return samples


# ── Training ─────────────────────────────────────────────────

def train(args):
    try:
        from trl import GRPOTrainer, GRPOConfig
        try:
            from unsloth import FastLanguageModel
            USE_UNSLOTH = True
        except ImportError:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            USE_UNSLOTH = False
    except ImportError:
        print("Install training deps: pip install trl transformers torch unsloth")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading model: {args.model} (Unsloth optimized: {USE_UNSLOTH})")
    
    if USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            use_gradient_checkpointing="unsloth",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Allow sanity runs on non-chat models (e.g., tiny GPT2) by supplying
    # a minimal template that concatenates role/content pairs.
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] }}: {{ message['content'] }}\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}"
        )

    # Curriculum: easy → medium → hard
    curriculum = args.curriculum.split(",") if args.curriculum else [args.task]

    for task_level in curriculum:
        print(f"\n{'='*50}")
        print(f"Training on task: {task_level}")
        print(f"{'='*50}")

        dataset = generate_dataset(task_level, n_samples=args.samples)
        reward_fn = make_reward_fn(task_level)

        config = GRPOConfig(
            output_dir=f"./checkpoints/{task_level}",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=5e-6,
            max_completion_length=256,
            num_generations=4,          # GRPO group size
            temperature=0.8,
            logging_steps=10,
            save_steps=100,
            report_to="none",
            seed=args.seed,
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_fn,
            args=config,
            train_dataset=dataset,
        )

        trainer.train()
        print(f"Checkpoint saved: ./checkpoints/{task_level}")

    print("\nTraining complete.")
    if args.push_to_hub:
        model.push_to_hub(args.hub_repo)
        tokenizer.push_to_hub(args.hub_repo)
        print(f"Pushed to HuggingFace Hub: {args.hub_repo}")


# ── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLM on AI Support Envoy environment")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--task", default="easy",
                        choices=["easy", "medium", "hard", "chaos", "frontier"],
                        help="Single task level (ignored if --curriculum set)")
    parser.add_argument("--curriculum", default="easy,medium,hard",
                        help="Comma-separated curriculum order, e.g. easy,medium,hard")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples", type=int, default=500,
                        help="Training samples per task level")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global seed for deterministic runs")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", default="punith2001/openenv-customer-support-model")
    args = parser.parse_args()
    train(args)
