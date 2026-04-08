#!/usr/bin/env python3
"""OpenEnv server application for Hugging Face Space deployment"""

import os
import sys
from typing import Any, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
async def home():
    """Stunning landing page for the environment"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OpenEnv | Customer Support</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            :root { --p: #6366f1; --bg: #0f172a; --card: #1e293b; }
            body { font-family: 'Outfit', sans-serif; background: var(--bg); color: white; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; overflow: hidden; }
            .glass { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); padding: 3rem; border-radius: 2rem; text-align: center; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); width: 400px; position: relative; }
            .orb { position: absolute; width: 300px; height: 300px; background: radial-gradient(circle, var(--p) 0%, transparent 70%); filter: blur(50px); opacity: 0.2; z-index: -1; top: -150px; left: -150px; }
            h1 { font-weight: 600; font-size: 2.5rem; margin-bottom: 0.5rem; background: linear-gradient(to right, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            p { color: #94a3b8; line-height: 1.6; }
            .badge { background: rgba(99, 102, 241, 0.2); color: #818cf8; padding: 0.5rem 1rem; border-radius: 1rem; font-size: 0.8rem; font-weight: 600; display: inline-block; margin-bottom: 1.5rem; }
            .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 2rem; }
            .stat { background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 1rem; border: 1px solid rgba(255,255,255,0.05); }
            .stat-val { font-weight: 600; font-size: 1.2rem; color: #fff; }
            .stat-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; }
            .live-pulse { width: 8px; height: 8px; background: #22c55e; border-radius: 50%; display: inline-block; margin-right: 5px; box-shadow: 0 0 10px #22c55e; animation: pulse 2s infinite; }
            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
        </style>
    </head>
    <body>
        <div class="glass">
            <div class="orb"></div>
            <div class="badge"><div class="live-pulse"></div> OPENENV COMPLIANT</div>
            <h1>Customer Support</h1>
            <p>A production-grade RL environment for intelligent ticket triage and resolution.</p>
            <div class="stat-grid">
                <div class="stat"><div class="stat-val">Healthy</div><div class="stat-label">Status</div></div>
                <div class="stat"><div class="stat-val">Active</div><div class="stat-label">Model API</div></div>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "customer-support-env"}

@app.post("/reset")
async def reset_environment(request: ResetRequest = ResetRequest()):
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
