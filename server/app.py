#!/usr/bin/env python3
"""OpenEnv server application for Hugging Face Space deployment"""

import os
import sys
from typing import Any, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.customer_support_env import CustomerSupportEnv, Action

# Create FastAPI app
app = FastAPI(title="Customer Support Environment", description="OpenEnv-compliant environment")

# Global environment instance
env = None
current_task = "easy"

class ResetRequest(BaseModel):
    task_level: str = "easy"

class StepRequest(BaseModel):
    action_type: str
    value: str
    reasoning: str = ""

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "customer-support-env"}

@app.post("/reset")
async def reset_environment(request: ResetRequest):
    """Reset environment and return initial observation"""
    global env, current_task
    
    try:
        current_task = request.task_level
        env = CustomerSupportEnv(task_level=current_task)
        observation = env.reset()
        
        return {
            "status": "ok",
            "observation": observation.model_dump(),
            "task": current_task
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step_environment(request: StepRequest):
    """Execute an action and return next state"""
    global env
    
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        # Create action from request
        action = Action(
            action_type=request.action_type,
            value=request.value,
            reasoning=request.reasoning
        )
        
        # Execute step
        observation, reward, done, info = env.step(action)
        
        return {
            "observation": observation.model_dump() if observation else None,
            "reward": float(reward),
            "done": bool(done),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
async def get_state():
    """Get current environment state"""
    global env
    
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized")
    
    try:
        state = env.state()
        return {"state": state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Entry point for running the server - this is what openenv expects"""
    import uvicorn
    
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting Customer Support Environment Server on {host}:{port}")
    print(f"Health check: http://{host}:{port}/health")
    print(f"Reset endpoint: http://{host}:{port}/reset")
    
    uvicorn.run(app, host=host, port=port)

# This is important - make sure the module can be run directly
if __name__ == "__main__":
    main()
