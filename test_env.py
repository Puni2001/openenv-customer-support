#!/usr/bin/env python3
"""Simple test to verify environment works"""

from src.customer_support_env import CustomerSupportEnv, Action

def test_environment():
    print("Testing Customer Support Environment...")
    
    # Test easy task
    env = CustomerSupportEnv("easy")
    obs = env.reset()
    
    print(f"✓ Environment initialized")
    print(f"  Task level: easy")
    print(f"  Tickets: {obs.tickets_remaining}")
    print(f"  First ticket: {obs.current_ticket.category.value}")
    
    # Test an action
    action = Action(action_type="categorize", value=obs.current_ticket.category.value)
    obs, reward, done, info = env.step(action)
    
    print(f"✓ Action executed")
    print(f"  Reward: {reward}")
    print(f"  Tickets remaining: {obs.tickets_remaining}")
    
    print("\n✅ Environment works correctly!")

if __name__ == "__main__":
    test_environment()
