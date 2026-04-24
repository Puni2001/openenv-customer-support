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
from typing import List, Dict

# ── Reward function (environment-grounded) ───────────────────

def make_reward_fn(task_level: str):
    """
    Returns a reward function compatible with TRL's GRPOTrainer.
    Each completion is scored by the environment's grader.
    """
    from src.customer_support_env import CustomerSupportEnv, Action
    from tasks.grader import TaskGrader

    def reward_fn(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
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

                # Run one environment step to get reward
                env = CustomerSupportEnv(task_level=task_level)
                env.reset()
                _, reward, _, _ = env.step(action)
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
        "easy": "You are a customer support triage agent. Classify the ticket. Respond in JSON: {\"action_type\": \"categorize\", \"value\": \"<category>\", \"reasoning\": \"<why>\"}",
        "medium": "You are a support ops agent. Set the correct priority. Respond in JSON: {\"action_type\": \"prioritize\", \"value\": \"<priority>\", \"reasoning\": \"<why>\"}",
        "hard": "You are a support resolution specialist. Resolve or escalate the ticket. Respond in JSON: {\"action_type\": \"resolve|escalate\", \"value\": \"<resolution>\", \"reasoning\": \"<why>\"}",
        "chaos": "You are handling a ticket storm. Resolve or escalate efficiently. Respond in JSON: {\"action_type\": \"resolve|escalate\", \"value\": \"<resolution>\", \"reasoning\": \"<why>\"}",
    }

    system = SYSTEM_PROMPTS.get(task_level, SYSTEM_PROMPTS["hard"])

    for _ in range(n_samples):
        obs = env.reset()
        ticket = obs.current_ticket
        if not ticket:
            continue

        kb = KNOWLEDGE_BASE.get(ticket.category.value, {})
        steps = ", ".join(kb.get("steps", []))

        user = f"""Ticket: {ticket.description}
Category: {ticket.category.value}
Sentiment: {ticket.sentiment:.2f}
Priority: {ticket.priority.value}
SLA Status: {obs.current_sla_status}
VIP: {ticket.is_vip}
Previous Contacts: {ticket.previous_contacts}
KB Steps: {steps}"""

        samples.append({
            "prompt": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "task_level": task_level
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
                        choices=["easy", "medium", "hard", "chaos"],
                        help="Single task level (ignored if --curriculum set)")
    parser.add_argument("--curriculum", default="easy,medium,hard",
                        help="Comma-separated curriculum order, e.g. easy,medium,hard")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples", type=int, default=500,
                        help="Training samples per task level")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", default="punith2001/ai-support-envoy-model")
    args = parser.parse_args()
    train(args)
