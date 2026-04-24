#!/usr/bin/env python3
"""
OpenEnv FastAPI Server — AI Support Envoy
==========================================
Endpoints:
  GET  /          — Dashboard UI
  GET  /health    — Health check
  POST /reset     — Start new episode
  POST /step      — Execute action
  GET  /state     — Current state
  GET  /metrics   — Aggregated metrics across sessions
"""

import os
import sys
import uuid
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.customer_support_env import CustomerSupportEnv, Action

app = FastAPI(
    title="AI Support Envoy — OpenEnv",
    description="Multi-level customer support RL environment with multi-agent and chaos modes",
    version="2.0.0"
)

# Mount static files
os.makedirs(os.path.join(os.path.dirname(__file__), "static", "runs"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Session-isolated environments (no global state collision)
_sessions: Dict[str, CustomerSupportEnv] = {}
_metrics: Dict[str, list] = {"scores": [], "rewards": [], "tasks": []}

VALID_TASK_LEVELS = {"easy", "medium", "hard", "chaos", "multi_agent_triage", "multi_agent_resolver"}

# ── Request Models ───────────────────────────────────────────

class ResetRequest(BaseModel):
    task_level: str = "easy"
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action_type: str
    value: str
    reasoning: str = ""

class RunUploadRequest(BaseModel):
    title: str
    log: str
    image_base64: str = ""

# ── Dashboard ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return open(os.path.join(os.path.dirname(__file__), "dashboard.html")).read()

@app.get("/history", response_class=HTMLResponse)
async def history_page():
    try:
        return open(os.path.join(os.path.dirname(__file__), "history.html")).read()
    except FileNotFoundError:
        return HTMLResponse("history.html not found.", status_code=404)

@app.get("/ui", response_class=HTMLResponse)
async def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Support Envoy | OpenEnv</title>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
  <style>
    :root{--p:#6366f1;--g:#22c55e;--w:#f59e0b;--r:#ef4444;--bg:#0a0f1e;--card:#111827;--border:rgba(255,255,255,0.07)}
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Outfit',sans-serif;background:var(--bg);color:#e2e8f0;min-height:100vh;padding:2rem}
    .hero{text-align:center;padding:3rem 1rem 2rem}
    .badge{display:inline-flex;align-items:center;gap:6px;background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);color:#818cf8;padding:.4rem 1rem;border-radius:2rem;font-size:.75rem;font-weight:600;letter-spacing:.05em;margin-bottom:1.5rem}
    .dot{width:7px;height:7px;background:var(--g);border-radius:50%;box-shadow:0 0 8px var(--g);animation:pulse 2s infinite}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
    h1{font-size:clamp(2rem,5vw,3.5rem);font-weight:700;background:linear-gradient(135deg,#818cf8,#c084fc,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.75rem}
    .subtitle{color:#64748b;font-size:1.05rem;max-width:600px;margin:0 auto 2.5rem}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;max-width:900px;margin:0 auto 3rem}
    .card{background:var(--card);border:1px solid var(--border);border-radius:1rem;padding:1.5rem;transition:border-color .2s}
    .card:hover{border-color:rgba(99,102,241,0.4)}
    .card-label{font-size:.7rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem}
    .card-val{font-size:1.4rem;font-weight:600;color:#f1f5f9}
    .card-sub{font-size:.75rem;color:#64748b;margin-top:.25rem}
    .levels{display:flex;flex-wrap:wrap;gap:.75rem;justify-content:center;margin-bottom:3rem}
    .level{background:var(--card);border:1px solid var(--border);border-radius:.75rem;padding:.75rem 1.25rem;font-size:.85rem}
    .level span{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:.5rem}
    .easy span{background:#22c55e}.medium span{background:#f59e0b}.hard span{background:#ef4444}.chaos span{background:#a855f7}.multi span{background:#38bdf8}
    .endpoints{max-width:700px;margin:0 auto;background:var(--card);border:1px solid var(--border);border-radius:1rem;overflow:hidden}
    .ep-row{display:flex;align-items:center;gap:1rem;padding:1rem 1.5rem;border-bottom:1px solid var(--border)}
    .ep-row:last-child{border-bottom:none}
    .method{font-size:.7rem;font-weight:700;padding:.25rem .6rem;border-radius:.4rem;min-width:48px;text-align:center}
    .get{background:rgba(34,197,94,.15);color:#22c55e}.post{background:rgba(99,102,241,.15);color:#818cf8}
    .ep-path{font-family:monospace;font-size:.9rem;color:#e2e8f0}
    .ep-desc{font-size:.8rem;color:#64748b;margin-left:auto}
    footer{text-align:center;color:#334155;font-size:.8rem;margin-top:3rem}
  </style>
</head>
<body>
  <div class="hero">
    <div class="badge"><div class="dot"></div>OPENENV COMPLIANT · v2.0</div>
    <h1>AI Support Envoy</h1>
    <p class="subtitle">A production-grade RL environment for intelligent ticket triage, prioritization, and resolution — with multi-agent and chaos modes.</p>
  </div>

  <div class="grid">
    <div class="card"><div class="card-label">Status</div><div class="card-val" style="color:#22c55e">Healthy</div><div class="card-sub">All systems operational</div></div>
    <div class="card"><div class="card-label">Theme</div><div class="card-val">3.1 + 1</div><div class="card-sub">World Modeling + Multi-Agent</div></div>
    <div class="card"><div class="card-label">Task Levels</div><div class="card-val">6</div><div class="card-sub">easy → chaos → multi-agent</div></div>
    <div class="card"><div class="card-label">Reward Model</div><div class="card-val">Dense</div><div class="card-sub">Shaped + SLA urgency signal</div></div>
  </div>

  <div class="levels">
    <div class="level easy"><span></span>Easy — Triage</div>
    <div class="level medium"><span></span>Medium — Prioritization</div>
    <div class="level hard"><span></span>Hard — Full Resolution</div>
    <div class="level chaos"><span></span>Chaos — Ticket Storm</div>
    <div class="level multi"><span></span>Multi-Agent Triage</div>
    <div class="level multi"><span></span>Multi-Agent Resolver</div>
  </div>

  <div class="endpoints">
    <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/health</span><span class="ep-desc">Health check</span></div>
    <div class="ep-row"><span class="method post">POST</span><span class="ep-path">/reset</span><span class="ep-desc">Start new episode</span></div>
    <div class="ep-row"><span class="method post">POST</span><span class="ep-path">/step</span><span class="ep-desc">Execute action</span></div>
    <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/state</span><span class="ep-desc">Current state</span></div>
    <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/metrics</span><span class="ep-desc">Aggregated metrics</span></div>
    <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/docs</span><span class="ep-desc">Interactive API docs</span></div>
  </div>

  <footer>AI Support Envoy · OpenEnv Hackathon · Theme #3.1 World Modeling + Theme #1 Multi-Agent</footer>
</body>
</html>"""

# ── API Endpoints ────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "ai-support-envoy", "version": "2.0.0",
            "active_sessions": len(_sessions)}

import time
import json
import glob
import base64

@app.post("/api/runs")
async def save_run(request: RunUploadRequest):
    timestamp = int(time.time())
    run_id = f"run_{timestamp}_{uuid.uuid4().hex[:6]}"
    run_dir = os.path.join(os.path.dirname(__file__), "static", "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    img_path = ""
    if request.image_base64:
        try:
            if "," in request.image_base64:
                header, encoded = request.image_base64.split(",", 1)
            else:
                encoded = request.image_base64
            data = base64.b64decode(encoded)
            img_filename = "reward_curve.png"
            with open(os.path.join(run_dir, img_filename), "wb") as f:
                f.write(data)
            img_path = f"/static/runs/{run_id}/{img_filename}"
        except Exception as e:
            print(f"Error decoding image: {e}")
            
    meta = {
        "id": run_id,
        "title": request.title,
        "log": request.log,
        "timestamp": timestamp,
        "image_path": img_path
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
        
    return {"status": "ok", "run_id": run_id}

@app.get("/api/runs")
async def list_runs():
    runs_dir = os.path.join(os.path.dirname(__file__), "static", "runs")
    meta_files = glob.glob(os.path.join(runs_dir, "*", "meta.json"))
    runs = []
    for m in meta_files:
        try:
            with open(m, "r") as f:
                runs.append(json.load(f))
        except:
            pass
    runs.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return runs

@app.post("/reset")
async def reset(request: ResetRequest, x_session_id: Optional[str] = Header(default=None)):
    if request.task_level not in VALID_TASK_LEVELS:
        raise HTTPException(400, f"task_level must be one of {sorted(VALID_TASK_LEVELS)}")

    session_id = x_session_id or str(uuid.uuid4())
    try:
        env = CustomerSupportEnv(task_level=request.task_level, seed=request.seed)
        obs = env.reset()
        _sessions[session_id] = env
        return {"status": "ok", "session_id": session_id,
                "observation": obs.model_dump(), "task": request.task_level}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/step")
async def step(request: StepRequest, x_session_id: Optional[str] = Header(default=None)):
    session_id = x_session_id or ""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(400, "No active session. Call POST /reset first with X-Session-Id header.")

    try:
        action = Action(action_type=request.action_type, value=request.value, reasoning=request.reasoning)
        obs, reward, done, info = env.step(action)

        if done:
            _metrics["scores"].append(env.state().get("cumulative_reward", 0))
            _metrics["tasks"].append(env.task_level)

        return {"observation": obs.model_dump() if obs else None,
                "reward": float(reward), "done": bool(done), "info": info}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/state")
async def get_state(x_session_id: Optional[str] = Header(default=None)):
    env = _sessions.get(x_session_id or "")
    if env is None:
        raise HTTPException(400, "No active session.")
    return {"state": env.state()}

@app.get("/demo")
async def demo_episode(task_level: str = "hard", use_llm: bool = False):
    """
    Run a simulated episode with a rule-based agent (no LLM needed).
    Used by the dashboard UI to show live step-by-step progress.
    """
    if task_level not in VALID_TASK_LEVELS:
        raise HTTPException(400, f"task_level must be one of {sorted(VALID_TASK_LEVELS)}")

    from src.customer_support_env import KNOWLEDGE_BASE, TicketCategory
    import random

    env = CustomerSupportEnv(task_level=task_level, seed=random.randint(0, 9999))
    obs = env.reset()
    steps = []

    agent_live = None
    if use_llm:
        import os
        from inference import SupportAgent
        api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise HTTPException(500, "HF_TOKEN not found in .env. Cannot use live LLM.")
        agent_live = SupportAgent(model_name=model, api_key=hf_token, base_url=api_base)

    while not env.done:
        t = obs.current_ticket
        if not t:
            break

        if use_llm and len(steps) >= env.task_config.get("max_steps", 15):
            break

        if use_llm and agent_live:
            action = agent_live.get_action(obs, task_level)
        else:
            kb = KNOWLEDGE_BASE.get(t.category.value, {})
            kb_steps = ", ".join(kb.get("steps", []))
            should_escalate = t.sentiment <= -0.7 or t.category == TicketCategory.COMPLAINT

            if task_level in ("easy", "multi_agent_triage"):
                if task_level == "multi_agent_triage" and should_escalate:
                    action = Action(action_type="escalate", value=f"Escalating: sentiment {t.sentiment:.2f}", reasoning="High severity routing")
                else:
                    action = Action(action_type="categorize", value=t.category.value,
                                    reasoning=f"Identified as {t.category.value} based on ticket content")
            elif task_level == "medium":
                from tasks.grader import TaskGrader
                pri = TaskGrader._expected_priority(t)
                action = Action(action_type="prioritize", value=pri.value,
                                reasoning=f"Sentiment {t.sentiment:.2f}, VIP={t.is_vip} → {pri.value}")
            else: # hard, chaos, multi_agent_resolver
                if should_escalate:
                    action = Action(action_type="escalate",
                                    value=f"Escalating: sentiment {t.sentiment:.2f}, category {t.category.value}",
                                    reasoning="High severity — requires senior team")
                else:
                    action = Action(action_type="resolve",
                                    value=f"Apologize for the inconvenience. {kb_steps}. Following up to confirm.",
                                    reasoning="KB-aligned resolution with empathy")

        obs_next, reward, done, info = env.step(action)

        steps.append({
            "ticket": {
                "id": t.id,
                "description": t.description,
                "category": t.category.value,
                "sentiment": round(t.sentiment, 2),
                "priority": t.priority.value,
                "sla_status": obs.current_sla_status,
                "is_vip": t.is_vip,
                "previous_contacts": t.previous_contacts,
            },
            "action": {
                "type": action.action_type,
                "value": action.value,
                "reasoning": action.reasoning,
            },
            "reward": round(reward, 4),
            "reward_breakdown": info.get("reward_breakdown", {}),
            "done": done,
        })
        obs = obs_next

    return {
        "task_level": task_level,
        "total_tickets": len(env.tickets),
        "total_reward": round(sum(s["reward"] for s in steps), 4),
        "steps": steps,
    }

@app.get("/metrics")
async def get_metrics():
    scores = _metrics["scores"]
    return {
        "total_episodes": len(scores),
        "avg_cumulative_reward": round(sum(scores) / max(1, len(scores)), 4),
        "max_reward": round(max(scores, default=0), 4),
        "active_sessions": len(_sessions),
        "task_distribution": {t: _metrics["tasks"].count(t) for t in set(_metrics["tasks"])}
    }

def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting AI Support Envoy on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
