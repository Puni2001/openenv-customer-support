#!/usr/bin/env python3
"""
Reproducible baseline vs trained evaluation for AI Support Envoy.
Writes machine-readable results to results/baseline_vs_trained.json.
"""

import argparse
import json
import os
import random
from typing import Dict, List

import numpy as np

from inference import SupportAgent
from src.customer_support_env import CustomerSupportEnv
from tasks.grader import TaskGrader


def run_task(model_name: str, task_level: str, seed: int, api_base: str, api_key: str) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    env = CustomerSupportEnv(task_level=task_level, seed=seed)
    obs = env.reset()
    agent = SupportAgent(model_name=model_name, api_key=api_key, base_url=api_base)

    actions: List[Dict] = []
    rewards: List[float] = []
    step = 0
    max_steps = env.task_config["max_steps"]

    while not env.done and step < max_steps:
        step += 1
        action = agent.get_action(obs, task_level)
        rec: Dict = {"reasoning": action.reasoning or ""}
        if action.action_type == "categorize":
            rec["categorization"] = action.value
        elif action.action_type == "prioritize":
            rec["priority"] = action.value
            rec["categorization"] = obs.current_ticket.category.value if obs.current_ticket else ""
        elif action.action_type in ("resolve", "escalate"):
            rec["resolution"] = action.value
            rec["escalated"] = action.action_type == "escalate"
            rec["resolved_within_sla"] = obs.current_sla_status != "breached"
        actions.append(rec)
        obs, reward, _, _ = env.step(action)
        rewards.append(float(reward))

    graders = {
        "easy": TaskGrader.grade_easy,
        "medium": TaskGrader.grade_medium,
        "hard": TaskGrader.grade_hard,
        "chaos": TaskGrader.grade_chaos,
    }
    score = graders[task_level](actions, env.tickets)
    return {"score": float(score), "avg_reward": float(sum(rewards) / max(1, len(rewards))), "steps": step}


def mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def evaluate_model(model_name: str, tasks: List[str], seeds: List[int], api_base: str, api_key: str) -> Dict:
    out: Dict = {"model_name": model_name, "tasks": {}}
    for task in tasks:
        print(f"[eval] model={model_name} task={task} seeds={seeds}", flush=True)
        runs = []
        for seed in seeds:
            print(f"[run:start] model={model_name} task={task} seed={seed}", flush=True)
            result = run_task(model_name, task, seed, api_base, api_key)
            runs.append(result)
            print(
                f"[run:done] model={model_name} task={task} seed={seed} "
                f"score={result['score']:.4f} avg_reward={result['avg_reward']:.4f} steps={result['steps']}",
                flush=True,
            )
        out["tasks"][task] = {
            "runs": runs,
            "mean_score": mean([r["score"] for r in runs]),
            "mean_avg_reward": mean([r["avg_reward"] for r in runs]),
            "mean_steps": mean([r["steps"] for r in runs]),
        }
        print(
            f"[task:summary] model={model_name} task={task} "
            f"mean_score={out['tasks'][task]['mean_score']:.4f} "
            f"mean_avg_reward={out['tasks'][task]['mean_avg_reward']:.4f}",
            flush=True,
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs trained model.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--trained-model", required=True)
    parser.add_argument("--tasks", default="easy,medium,hard")
    parser.add_argument("--seeds", default="41,42,43,44,45")
    parser.add_argument("--output", default="results/baseline_vs_trained.json")
    args = parser.parse_args()

    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise RuntimeError("HF_TOKEN is required")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(f"[config] tasks={tasks} seeds={seeds}", flush=True)
    print(f"[config] base_model={args.base_model}", flush=True)
    print(f"[config] trained_model={args.trained_model}", flush=True)

    base_eval = evaluate_model(args.base_model, tasks, seeds, api_base, api_key)
    trained_eval = evaluate_model(args.trained_model, tasks, seeds, api_base, api_key)

    merged = {"tasks": {}}
    for task in tasks:
        b = base_eval["tasks"][task]
        t = trained_eval["tasks"][task]
        merged["tasks"][task] = {
            "base_mean_score": b["mean_score"],
            "trained_mean_score": t["mean_score"],
            "delta_score": t["mean_score"] - b["mean_score"],
            "base_mean_avg_reward": b["mean_avg_reward"],
            "trained_mean_avg_reward": t["mean_avg_reward"],
            "delta_avg_reward": t["mean_avg_reward"] - b["mean_avg_reward"],
        }

    payload = {
        "config": {"tasks": tasks, "seeds": seeds},
        "base": base_eval,
        "trained": trained_eval,
        "summary": merged,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[done] wrote evaluation: {args.output}", flush=True)


if __name__ == "__main__":
    main()
