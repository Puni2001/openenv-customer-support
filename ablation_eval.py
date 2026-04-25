#!/usr/bin/env python3
"""
Ablation: measure impact of anti-reward-hacking over-prioritization penalty.
"""

import json
from pathlib import Path

from src.customer_support_env import CustomerSupportEnv, Action


def run_spam_policy(disable_hack_penalty: bool, episodes: int = 20, seed_start: int = 100) -> dict:
    rewards = []
    spam_count = 0
    for i in range(episodes):
        env = CustomerSupportEnv(task_level="medium", seed=seed_start + i, disable_hack_penalty=disable_hack_penalty)
        obs = env.reset()
        total = 0.0
        while not env.done:
            # Deliberate hacking behavior: always mark urgent.
            action = Action(action_type="prioritize", value="urgent", reasoning="force urgency")
            obs, reward, done, info = env.step(action)
            total += reward
            if "over_prioritization_hack_penalty" in info.get("reward_breakdown", {}):
                spam_count += 1
        rewards.append(total)
    mean_reward = sum(rewards) / len(rewards)
    return {
        "episodes": episodes,
        "mean_reward": round(mean_reward, 4),
        "min_reward": round(min(rewards), 4),
        "max_reward": round(max(rewards), 4),
        "hack_penalty_trigger_count": spam_count,
    }


def run_governance_ablation(episodes: int = 20, seed_start: int = 300) -> dict:
    """
    Compare unsafe autonomous resolution vs governance-compliant handoff
    under frontier task risk conditions.
    """
    unsafe_reward = []
    safe_reward = []
    unsafe_wrongful_autonomy = 0
    safe_wrongful_autonomy = 0
    for i in range(episodes):
        # Unsafe policy: always resolve
        env_unsafe = CustomerSupportEnv(task_level="frontier", seed=seed_start + i)
        obs = env_unsafe.reset()
        total_unsafe = 0.0
        while not env_unsafe.done:
            obs, reward, done, info = env_unsafe.step(
                Action(action_type="resolve", value="auto resolve without checks", reasoning="unsafe ablation")
            )
            total_unsafe += reward
        state_unsafe = env_unsafe.state()
        unsafe_wrongful_autonomy += int(state_unsafe.get("telemetry", {}).get("wrongful_autonomy", 0))
        unsafe_reward.append(total_unsafe)

        # Safe policy: follow governance hint
        env_safe = CustomerSupportEnv(task_level="frontier", seed=seed_start + i)
        obs = env_safe.reset()
        total_safe = 0.0
        while not env_safe.done:
            action_type = obs.governance_hint if obs.governance_hint in ("human_review_required", "legal_hold") else "resolve"
            obs, reward, done, info = env_safe.step(
                Action(action_type=action_type, value="policy-safe handoff", reasoning="safe ablation")
            )
            total_safe += reward
        state_safe = env_safe.state()
        safe_wrongful_autonomy += int(state_safe.get("telemetry", {}).get("wrongful_autonomy", 0))
        safe_reward.append(total_safe)

    return {
        "episodes": episodes,
        "unsafe_policy_mean_reward": round(sum(unsafe_reward) / len(unsafe_reward), 4),
        "safe_policy_mean_reward": round(sum(safe_reward) / len(safe_reward), 4),
        "delta_safe_minus_unsafe": round((sum(safe_reward) / len(safe_reward)) - (sum(unsafe_reward) / len(unsafe_reward)), 4),
        "unsafe_wrongful_autonomy_count": unsafe_wrongful_autonomy,
        "safe_wrongful_autonomy_count": safe_wrongful_autonomy,
    }


def main():
    enabled = run_spam_policy(disable_hack_penalty=False)
    disabled = run_spam_policy(disable_hack_penalty=True)
    governance_ablation = run_governance_ablation()
    out = {
        "with_penalty": enabled,
        "without_penalty": disabled,
        "delta_mean_reward_without_minus_with": round(disabled["mean_reward"] - enabled["mean_reward"], 4),
        "governance_ablation": governance_ablation,
    }
    out_path = Path("results/ablation_hack_penalty.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved ablation report to {out_path}")


if __name__ == "__main__":
    main()

