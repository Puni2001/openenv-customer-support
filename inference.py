#!/usr/bin/env python3
"""
STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import time
import textwrap
from dotenv import load_dotenv
from typing import List, Optional
from openai import OpenAI

# Load environment variables
load_dotenv(override=True)

from src.customer_support_env import CustomerSupportEnv, Action
from tasks.grader import TaskGrader

# --- MANDATORY CONFIGURATION (AS PER HACKATHON CHECKLIST) ---
# Defaults are set ONLY for API_BASE_URL and MODEL_NAME
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
# No default for HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN")

BENCHMARK = "customer-support-env"
MAX_STEPS = 15
TEMPERATURE = 0.2

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = str(action).replace("\n", " ").replace("\t", " ")[:60]
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_val = str(success).lower()
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

class SupportAgent:
    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model_name = model_name
        # MANDATORY: Using the OpenAI client for all LLM calls
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def get_action(self, observation, task_level: str) -> Action:
        ticket = observation.current_ticket
        if not ticket:
            return Action(action_type="request_info", value="No tickets")
        
        system = f"""
        Expert customer support agent. Respond ONLY in valid JSON.
        
        STRICT RULES FOR LEVEL: {task_level.upper()}
        - If Level is EASY: action_type MUST be "categorize" (technical|billing|account|feature_request|complaint).
        - If Level is MEDIUM: action_type MUST be "prioritize" (low|medium|high|urgent).
        - If Level is HARD: action_type MUST be "resolve" or "escalate".
        
        Format: {{ "action_type": "...", "value": "...", "reasoning": "..." }}
        """.strip()
        user = f"Ticket: {ticket.description}. Level: {task_level}."

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=TEMPERATURE,
                    max_tokens=200
                )
                content = resp.choices[0].message.content
                
                # Extract JSON
                if "```json" in content: content = content.split("```json")[1].split("```")[0]
                elif "```" in content: content = content.split("```")[1].split("```")[0]
                
                data = json.loads(content.strip())
                return Action(
                    action_type=data.get("action_type", "request_info"),
                    value=data.get("value", ""),
                    reasoning=data.get("reasoning", "")
                )
            except Exception as e:
                # Basic rate limit handling
                if "429" in str(e):
                    time.sleep((attempt + 1) * 10)
                    continue
                if attempt == 2: return Action(action_type="request_info", value=str(e)[:50])
                time.sleep(1)
        
        return Action(action_type="request_info", value="Max retries reached")

def run_task(task_level: str):
    log_start(task_level, BENCHMARK, MODEL_NAME)
    env = CustomerSupportEnv(task_level=task_level)
    agent = SupportAgent(MODEL_NAME, HF_TOKEN, API_BASE_URL)
    
    obs = env.reset()
    rewards, actions, step = [], [], 0
    
    while not env.done and step < MAX_STEPS:
        step += 1
        action = agent.get_action(obs, task_level)
        
        rec = {}
        if action.action_type == "categorize": rec["categorization"] = action.value
        elif action.action_type == "prioritize": rec["priority"] = action.value
        elif action.action_type == "resolve": rec["resolution"] = action.value
        elif action.action_type == "escalate": 
            rec["escalated"] = True
            rec["resolution"] = action.value
        
        actions.append(rec)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        log_step(step, f"{action.action_type}({action.value})", reward, done, None)
    
    score = 0.0
    if task_level == "easy": score = TaskGrader.grade_easy(actions, env.tickets)
    elif task_level == "medium": score = TaskGrader.grade_medium(actions, env.tickets)
    elif task_level == "hard": score = TaskGrader.grade_hard(actions, env.tickets, env.knowledge_base)
    
    log_end(env.done, step, score, rewards)

def main():
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set.")
        return
    for task in ["easy", "medium", "hard"]:
        run_task(task)

if __name__ == "__main__":
    main()
