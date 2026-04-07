#!/usr/bin/env python3
"""Baseline inference script for Customer Support Environment"""

import os
import json
import time
from typing import Dict, List
from openai import OpenAI
from src.customer_support_env import CustomerSupportEnv, Action

# Environment variables (required for submission)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# Configuration
MAX_STEPS = 30
TEMPERATURE = 0.2

class SupportAgent:
    """Agent that uses OpenAI to handle customer support tickets"""
    
    def __init__(self):
        if not API_KEY:
            raise ValueError("API_KEY not set! Set HF_TOKEN or OPENAI_API_KEY")
        
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        print(f"✓ Agent initialized with model: {MODEL_NAME}")
    
    def get_action(self, observation, task_level: str) -> Action:
        """Get action decision from LLM"""
        
        # Build prompt with context
        ticket = observation.current_ticket
        
        prompt = f"""
You are a customer support agent. Handle the following ticket:

=== TICKET INFO ===
ID: {ticket.id}
Customer: {ticket.customer_id}
Category: {ticket.category.value}
Description: {ticket.description}
Sentiment: {ticket.sentiment:.2f} (negative = angry, positive = happy)
Current Priority: {ticket.priority.value}

=== CONTEXT ===
Task Level: {task_level} ({self._get_task_description(task_level)})
Tickets Remaining: {observation.tickets_remaining}
Tickets Handled: {observation.tickets_handled}
Recent Actions: {observation.recent_actions[-3:] if observation.recent_actions else "None"}

=== AVAILABLE ACTIONS ===
1. categorize: Set the correct ticket category
   - Options: technical, billing, account, feature_request, complaint
   
2. prioritize: Set priority level (for medium/hard tasks)
   - Options: low, medium, high, urgent
   
3. resolve: Provide solution and close ticket
   - Value: your resolution notes
   
4. escalate: Escalate to manager (for complex issues)
   - Value: reason for escalation
   
5. request_info: Ask customer for more details
   - Value: what information you need

=== INSTRUCTIONS ===
Based on the ticket, choose the NEXT BEST ACTION.
For easy task: focus on categorization
For medium task: categorize + set priority
For hard task: aim to resolve or escalate

Respond with JSON only:
{{"action_type": "categorize", "value": "technical", "reasoning": "brief reason"}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful customer support agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=150
            )
            
            action_text = response.choices[0].message.content
            return self._parse_action(action_text)
            
        except Exception as e:
            print(f"  ⚠️  LLM error: {e}")
            return self._fallback_action(ticket, task_level)
    
    def _get_task_description(self, task_level: str) -> str:
        """Get description of current task"""
        descriptions = {
            "easy": "Just categorize the ticket correctly",
            "medium": "Categorize AND set appropriate priority",
            "hard": "Resolve the ticket using knowledge base or escalate if needed"
        }
        return descriptions.get(task_level, "Handle ticket appropriately")
    
    def _parse_action(self, text: str) -> Action:
        """Parse JSON response into Action object"""
        try:
            # Extract JSON from response
            if "{" in text and "}" in text:
                json_str = text[text.find("{"):text.rfind("}")+1]
                data = json.loads(json_str)
                return Action(
                    action_type=data.get("action_type", "request_info"),
                    value=data.get("value", ""),
                    reasoning=data.get("reasoning")
                )
        except Exception as e:
            print(f"  ⚠️  Parse error: {e}")
        
        # Fallback
        return Action(action_type="request_info", value="Need more information")
    
    def _fallback_action(self, ticket, task_level: str) -> Action:
        """Simple rule-based fallback when LLM fails"""
        if task_level == "easy":
            # Just categorize based on keywords
            desc_lower = ticket.description.lower()
            if any(word in desc_lower for word in ["charge", "billing", "payment", "refund"]):
                return Action(action_type="categorize", value="billing")
            elif any(word in desc_lower for word in ["error", "crash", "slow", "login"]):
                return Action(action_type="categorize", value="technical")
            elif any(word in desc_lower for word in ["password", "account", "locked"]):
                return Action(action_type="categorize", value="account")
            else:
                return Action(action_type="categorize", value="feature_request")
        else:
            # For harder tasks, try to resolve
            return Action(action_type="resolve", value="Checking this issue for you")

def run_episode(task_level: str) -> Dict:
    """Run one complete episode and return results"""
    
    print(f"\n{'='*60}")
    print(f"🎯 TASK: {task_level.upper()}")
    print(f"{'='*60}")
    
    env = CustomerSupportEnv(task_level=task_level)
    agent = SupportAgent()
    
    observation = env.reset()
    total_reward = 0.0
    actions_taken = []
    step = 0
    
    print(f"📋 Starting with {observation.tickets_remaining} tickets")
    
    while not env.done and step < MAX_STEPS:
        step += 1
        
        # Get action from agent
        action = agent.get_action(observation, task_level)
        
        # Execute action
        observation, reward, done, info = env.step(action)
        total_reward += reward
        
        # Record action
        actions_taken.append({
            "step": step,
            "action": action.action_type,
            "value": action.value,
            "reward": reward
        })
        
        # Print progress
        status = "✅" if reward > 0 else "⚠️" if reward < 0 else "➡️"
        print(f"{status} Step {step:2d}: {action.action_type:12s} -> {action.value[:40]:40s} | reward: {reward:+.2f}")
        
        if done:
            print(f"\n✨ Episode complete! Total reward: {total_reward:.2f}")
            break
    
    return {
        "task": task_level,
        "total_reward": total_reward,
        "steps": step,
        "tickets_handled": observation.tickets_handled,
        "actions": actions_taken
    }

def main():
    """Run all three tasks and report scores"""
    
    print("\n" + "="*60)
    print("🚀 CUSTOMER SUPPORT ENVIRONMENT - BASELINE EVALUATION")
    print("="*60)
    
    # Check credentials
    if not API_KEY:
        print("\n❌ ERROR: API_KEY not set!")
        print("\nPlease set your API key:")
        print("  export HF_TOKEN=your_token_here")
        print("  or")
        print("  export OPENAI_API_KEY=your_key_here")
        print("\n⚠️  Note: For Hugging Face, use HF_TOKEN")
        return
    
    print(f"\n🔧 Configuration:")
    print(f"   Model: {MODEL_NAME}")
    print(f"   API: {API_BASE_URL}")
    print(f"   Max steps: {MAX_STEPS}")
    
    # Run all difficulty levels
    results = {}
    for difficulty in ["easy", "medium", "hard"]:
        try:
            print(f"\n{'='*60}")
            print(f"📝 Running {difficulty.upper()} task...")
            result = run_episode(difficulty)
            results[difficulty] = result
            time.sleep(1)  # Small delay between runs
        except Exception as e:
            print(f"\n❌ Error running {difficulty}: {e}")
            results[difficulty] = {"total_reward": 0.0, "error": str(e)}
    
    # Print summary
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    for difficulty, result in results.items():
        reward = result.get("total_reward", 0)
        steps = result.get("steps", 0)
        tickets = result.get("tickets_handled", 0)
        
        # Normalize reward to 0-1 scale for grader
        normalized = min(max(reward / 5.0, 0), 1)  # Rough normalization
        
        print(f"\n{difficulty.upper()}:")
        print(f"  Score: {normalized:.3f} / 1.0")
        print(f"  Reward: {reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Tickets handled: {tickets}")
    
    # Save results
    with open("baseline_scores.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to baseline_scores.json")
    print("\n🏁 Evaluation complete!")

if __name__ == "__main__":
    main()
