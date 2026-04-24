#!/usr/bin/env python3
"""
AI Support Envoy — Inference Script
=====================================
STDOUT FORMAT (mandatory):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import time
import sys
from typing import List, Optional, Dict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

from src.customer_support_env import CustomerSupportEnv, Action, KNOWLEDGE_BASE
from tasks.grader import TaskGrader

# ── Config ──────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
BENCHMARK    = "customer-support-env"
TEMPERATURE  = 0.2

# ── Logging ─────────────────────────────────────────────────

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error or "null"
    act = str(action).replace("\n", " ")[:80]
    print(f"[STEP] step={step} action={act} reward={reward:.4f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)

# ── Agent ────────────────────────────────────────────────────

class SupportAgent:
    """LLM-powered support agent using HuggingFace Router (OpenAI-compatible)."""

    SYSTEM_PROMPTS = {
        "easy": """You are an expert customer support triage agent.
Your ONLY job: classify the ticket into exactly one category.
Valid categories: technical | billing | account | feature_request | complaint

Respond ONLY with valid JSON:
{"action_type": "categorize", "value": "<category>", "reasoning": "<brief explanation>"}""",

        "medium": """You are a senior support operations agent.
Your job: set the correct priority for this ticket based on sentiment, SLA, and VIP status.
Valid priorities: low | medium | high | urgent

Rules:
- sentiment <= -0.8 → urgent
- sentiment <= -0.5 → high  
- VIP customer → escalate one level
- feature_request → low (unless VIP)
- complaint with negative sentiment → high

Respond ONLY with valid JSON:
{"action_type": "prioritize", "value": "<priority>", "reasoning": "<brief explanation>"}""",

        "hard": """You are an expert customer support resolution specialist.
Your job: resolve the ticket using the knowledge base OR escalate if necessary.

Escalate when: sentiment <= -0.7, category is complaint, customer is VIP with negative sentiment, or 4+ previous contacts.

For resolution, include: diagnosis, steps taken, and empathetic language if sentiment is negative.
Use words like: apologize, sorry, understand, frustration, priority, immediately — for negative sentiment tickets.

Respond ONLY with valid JSON:
{"action_type": "resolve" OR "escalate", "value": "<resolution or escalation reason>", "reasoning": "<explanation>"}""",

        "chaos": """You are a high-performance support agent handling a ticket storm.
Prioritize: VIP customers, breached SLAs, urgent tickets first.
Resolve or escalate decisively. Be efficient — every step costs time.

Respond ONLY with valid JSON:
{"action_type": "resolve" OR "escalate", "value": "<resolution or escalation reason>", "reasoning": "<explanation>"}""",

        "multi_agent_triage": """You are the Triage Agent in a two-agent support pipeline.
Your job: categorize the ticket and route it to the correct resolver team.
Valid categories: technical | billing | account | feature_request | complaint

Respond ONLY with valid JSON:
{"action_type": "categorize", "value": "<category>", "reasoning": "<routing rationale>"}""",

        "multi_agent_resolver": """You are the Resolver Agent in a two-agent support pipeline.
You receive pre-triaged tickets. Your job: resolve or escalate.
The triage decision is provided in context.

Respond ONLY with valid JSON:
{"action_type": "resolve" OR "escalate", "value": "<resolution or escalation reason>", "reasoning": "<explanation>"}"""
    }

    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def get_action(self, observation, task_level: str) -> Action:
        ticket = observation.current_ticket
        if not ticket:
            return Action(action_type="request_info", value="no_ticket", reasoning="Queue empty")

        system = self.SYSTEM_PROMPTS.get(task_level, self.SYSTEM_PROMPTS["hard"])

        # Build rich user context
        kb_hint = ""
        if task_level in ("hard", "chaos", "multi_agent_resolver"):
            kb = KNOWLEDGE_BASE.get(ticket.category.value, {})
            steps = ", ".join(kb.get("steps", []))
            escalate_if = ", ".join(kb.get("escalate_if", []))
            kb_hint = f"\nKnowledge Base Steps: {steps}\nEscalate if: {escalate_if or 'N/A'}"

        user = f"""Ticket ID: {ticket.id}
Description: {ticket.description}
Category: {ticket.category.value}
Sentiment: {ticket.sentiment:.2f} (-1=very negative, +1=very positive)
Priority: {ticket.priority.value}
SLA Status: {observation.current_sla_status}
VIP Customer: {ticket.is_vip}
Previous Contacts: {ticket.previous_contacts}
Urgent tickets in queue: {observation.urgent_tickets_in_queue}
Agent Role: {observation.agent_role}{kb_hint}"""

        if observation.triage_decision:
            user += f"\nTriage Decision: {observation.triage_decision}"

        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=300
                )
                content = resp.choices[0].message.content.strip()

                # Extract JSON robustly
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                # Find JSON object
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    content = content[start:end]

                data = json.loads(content)
                return Action(
                    action_type=data.get("action_type", "request_info"),
                    value=str(data.get("value", "")),
                    reasoning=str(data.get("reasoning", ""))
                )

            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    wait = (attempt + 1) * 15
                    time.sleep(wait)
                    continue
                if attempt >= 3:
                    return Action(action_type="request_info", value="error", reasoning=err_str[:80])
                time.sleep(2 ** attempt)

        return Action(action_type="request_info", value="max_retries", reasoning="")

# ── Task Runner ──────────────────────────────────────────────

def run_task(task_level: str, agent: SupportAgent) -> Dict:
    log_start(task_level, BENCHMARK, MODEL_NAME)

    env = CustomerSupportEnv(task_level=task_level)
    obs = env.reset()
    rewards: List[float] = []
    actions: List[Dict] = []
    step = 0
    max_steps = env.task_config["max_steps"]

    while not env.done and step < max_steps:
        step += 1
        action = agent.get_action(obs, task_level)

        # Build action record for grader
        rec: Dict = {"reasoning": action.reasoning or ""}
        if action.action_type == "categorize":
            rec["categorization"] = action.value
        elif action.action_type == "prioritize":
            rec["priority"] = action.value
            rec["categorization"] = obs.current_ticket.category.value if obs.current_ticket else ""
        elif action.action_type == "resolve":
            rec["resolution"] = action.value
            rec["escalated"] = False
            rec["resolved_within_sla"] = obs.current_sla_status != "breached"
        elif action.action_type == "escalate":
            rec["escalated"] = True
            rec["resolution"] = action.value
            rec["resolved_within_sla"] = obs.current_sla_status != "breached"

        actions.append(rec)

        try:
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            log_step(step, f"{action.action_type}({action.value[:40]})", reward, done, None)
        except Exception as e:
            log_step(step, f"{action.action_type}(error)", 0.0, False, str(e)[:60])
            break

    # Grade
    score = 0.0
    graders = {
        "easy": lambda: TaskGrader.grade_easy(actions, env.tickets),
        "medium": lambda: TaskGrader.grade_medium(actions, env.tickets),
        "hard": lambda: TaskGrader.grade_hard(actions, env.tickets),
        "chaos": lambda: TaskGrader.grade_chaos(actions, env.tickets),
    }
    if task_level in graders:
        score = graders[task_level]()

    success = score >= 0.6
    log_end(success, step, score, rewards)

    return {"task": task_level, "score": score, "steps": step, "rewards": rewards, "success": success}

# ── Main ─────────────────────────────────────────────────────

def main():
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set.", file=sys.stderr)
        sys.exit(1)

    agent = SupportAgent(MODEL_NAME, HF_TOKEN, API_BASE_URL)
    results = []

    tasks = ["easy", "medium", "hard"]
    # Optionally run chaos mode
    if os.getenv("RUN_CHAOS", "false").lower() == "true":
        tasks.append("chaos")

    for task in tasks:
        result = run_task(task, agent)
        results.append(result)
        time.sleep(2)  # brief pause between tasks

    # Summary
    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        avg_r = sum(r["rewards"]) / max(1, len(r["rewards"]))
        print(f"[{status}] {r['task']:25s} score={r['score']:.4f}  avg_reward={avg_r:.4f}  steps={r['steps']}", flush=True)

if __name__ == "__main__":
    main()
