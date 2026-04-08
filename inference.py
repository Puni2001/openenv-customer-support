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

# Load environment variables
load_dotenv(override=True)

from typing import List, Optional, Dict
from openai import OpenAI
import google.generativeai as genai

from src.customer_support_env import CustomerSupportEnv, Action
from tasks.grader import TaskGrader

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-flash-latest")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")

if "gemini" in MODEL_NAME.lower():
    API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
else:
    API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")

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
        self.api_key = api_key
        self.base_url = base_url
        self.use_native = "gemini" in model_name.lower() and "generativelanguage" not in base_url.lower() or os.getenv("GEMINI_API_KEY")
        
        if self.use_native:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=model_name.replace("models/", ""),
                generation_config={"temperature": TEMPERATURE}
            )
        else:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def get_action(self, observation, task_level: str) -> Action:
        ticket = observation.current_ticket
        if not ticket:
            return Action(action_type="request_info", value="No tickets")
        
        system = textwrap.dedent(f"""
            Expert support agent. Respond in valid JSON only:
            {{ "action_type": "categorize|prioritize|resolve|escalate", "value": "...", "reasoning": "..." }}
            Easy: categorize (technical|billing|account|feature_request|complaint).
            Medium: prioritize (low|medium|high|urgent).
            Hard: resolve or escalate.
        """).strip()
        user = f"Ticket: {ticket.description}. Level: {task_level}."

        for attempt in range(3):
            try:
                if self.use_native:
                    resp = self.model.generate_content(f"{system}\n\n{user}")
                    content = resp.text
                else:
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
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    time.sleep((attempt + 1) * 10) # Simple wait
                    continue
                if attempt == 2: return Action(action_type="request_info", value=str(e)[:50])
                time.sleep(1)

def run_task(task_level: str):
    log_start(task_level, BENCHMARK, MODEL_NAME)
    env = CustomerSupportEnv(task_level=task_level)
    agent = SupportAgent(MODEL_NAME, API_KEY, API_BASE_URL)
    
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
    if not API_KEY: return
    for task in ["easy", "medium", "hard"]:
        run_task(task)

if __name__ == "__main__":
    main()
