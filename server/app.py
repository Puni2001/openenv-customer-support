#!/usr/bin/env python3
"""
OpenEnv FastAPI Server — FrontierOps Arena
===========================================
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

import time
import json
import glob
import base64
import random
import shutil
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.customer_support_env import CustomerSupportEnv, Action, KNOWLEDGE_BASE, TicketCategory
from src.agent import SupportAgent
from src.telemetry import aggregate_slo_kpi

app = FastAPI(
    title="FrontierOps Arena — OpenEnv",
    description="Multi-level enterprise operations RL environment with multi-agent and frontier modes",
    version="2.0.0"
)

# Mount static files
os.makedirs(os.path.join(os.path.dirname(__file__), "static", "runs"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Session-isolated environments (no global state collision)
_sessions: Dict[str, CustomerSupportEnv] = {}
_metrics: Dict[str, list] = {"scores": [], "rewards": [], "tasks": []}
_episode_telemetry: list = []

VALID_TASK_LEVELS = {"easy", "medium", "hard", "chaos", "multi_agent_triage", "multi_agent_resolver", "frontier"}

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


def _persist_run(title: str, log: str, image_bytes: Optional[bytes] = None) -> str:
    timestamp = int(time.time())
    run_id = f"run_{timestamp}_{uuid.uuid4().hex[:6]}"
    run_dir = os.path.join(os.path.dirname(__file__), "static", "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    img_path = ""
    if image_bytes:
        img_filename = "reward_curve.png"
        with open(os.path.join(run_dir, img_filename), "wb") as f:
            f.write(image_bytes)
        img_path = f"/static/runs/{run_id}/{img_filename}"

    meta = {
        "id": run_id,
        "title": title,
        "log": log,
        "timestamp": timestamp,
        "image_path": img_path
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return run_id


def _format_episode_log(task_level: str, source: str, cumulative_reward: float, state: Dict[str, Any]) -> str:
    telemetry = state.get("telemetry", {}) or {}
    lines = [
        f"[AUTO] source={source}",
        f"[AUTO] task_level={task_level}",
        f"[AUTO] tickets_handled={state.get('tickets_handled', 0)} / total_tickets={state.get('total_tickets', 0)}",
        f"[AUTO] steps={state.get('steps', 0)}",
        f"[AUTO] cumulative_reward={cumulative_reward:.4f}",
        f"[AUTO] safe_handoff={telemetry.get('safe_handoff', 0)}",
        f"[AUTO] unsafe_action_blocked={telemetry.get('unsafe_action_blocked', 0)}",
        f"[AUTO] wrongful_autonomy={telemetry.get('wrongful_autonomy', 0)}",
        f"[AUTO] governance_blocks={telemetry.get('governance_blocks', 0)}",
        f"[AUTO] tool_calls={telemetry.get('tool_calls', 0)}",
        f"[AUTO] tool_fallbacks={telemetry.get('tool_fallbacks', 0)}",
    ]
    return "\n".join(lines)

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
    # Keep legacy /ui route but serve the canonical dashboard to avoid stale duplicate UI.
    return open(os.path.join(os.path.dirname(__file__), "dashboard.html")).read()

# ── API Endpoints ────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "frontierops-arena", "version": "2.0.0",
            "active_sessions": len(_sessions)}

@app.post("/api/runs")
async def save_run(request: RunUploadRequest):
    image_bytes = None
    if request.image_base64:
        try:
            if "," in request.image_base64:
                _, encoded = request.image_base64.split(",", 1)
            else:
                encoded = request.image_base64
            image_bytes = base64.b64decode(encoded, validate=True)
        except Exception as e:
            print(f"Error decoding image: {e}")

    run_id = _persist_run(request.title, request.log, image_bytes=image_bytes)
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
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Skipping invalid run metadata file {m}: {exc}")
    runs.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return runs


@app.delete("/api/runs")
async def clear_runs():
    runs_dir = os.path.join(os.path.dirname(__file__), "static", "runs")
    removed = 0
    if os.path.isdir(runs_dir):
        for entry in os.listdir(runs_dir):
            entry_path = os.path.join(runs_dir, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path, ignore_errors=True)
                removed += 1
    return {"status": "ok", "removed_runs": removed}

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
            state = env.state()
            _metrics["scores"].append(state.get("cumulative_reward", 0))
            _metrics["tasks"].append(env.task_level)
            total_tickets = max(1, int(state.get("total_tickets", 1)))
            tickets_handled = int(state.get("tickets_handled", 0))
            telemetry = state.get("telemetry", {})
            episode_row = {
                "task_level": env.task_level,
                "cumulative_reward": float(state.get("cumulative_reward", 0.0)),
                "resolution_rate": float(tickets_handled / total_tickets),
                "escalation_rate": float(telemetry.get("safe_handoff", 0)) / total_tickets,
                "safe_handoff_rate": float(telemetry.get("safe_handoff", 0)) / total_tickets,
                "blocked_unsafe_action_rate": float(telemetry.get("unsafe_action_blocked", 0)) / total_tickets,
                "wrongful_autonomy_rate": float(telemetry.get("wrongful_autonomy", 0)) / total_tickets,
                "tool_calls_per_ticket": float(telemetry.get("tool_calls", 0)) / total_tickets,
                "tool_fallback_rate": float(telemetry.get("tool_fallbacks", 0)) / max(1, float(telemetry.get("tool_calls", 0))),
            }
            _episode_telemetry.append(episode_row)
            auto_title = f"Auto API Episode · {env.task_level} · {session_id[:8]}"
            auto_log = _format_episode_log(
                task_level=env.task_level,
                source="api_step",
                cumulative_reward=float(state.get("cumulative_reward", 0.0)),
                state=state,
            )
            _persist_run(auto_title, auto_log)

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
async def demo_episode(task_level: str = "hard", use_llm: bool = False, agent_type: str = "rule_based"):
    """
    Run a simulated episode with a rule-based agent (no LLM needed).
    Used by the dashboard UI to show live step-by-step progress.
    """
    if task_level not in VALID_TASK_LEVELS:
        raise HTTPException(400, f"task_level must be one of {sorted(VALID_TASK_LEVELS)}")

    env = CustomerSupportEnv(task_level=task_level, seed=random.randint(0, 9999))
    obs = env.reset()
    steps = []

    agent_live = None
    model_name = None
    if use_llm:
        api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        base_model = os.getenv("BASE_MODEL_NAME", os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"))
        trained_model = os.getenv("TRAINED_MODEL_NAME", base_model)
        if agent_type == "base_llm":
            model_name = base_model
        elif agent_type == "rl_llm":
            model_name = trained_model
        else:
            model_name = base_model
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise HTTPException(500, "HF_TOKEN not found in .env. Cannot use live LLM.")
        
        print(f"[DEMO] Running with agent_type={agent_type}, model={model_name}")
        agent_live = SupportAgent(model_name=model_name, api_key=hf_token, base_url=api_base)

    repeated_action_count = 0
    previous_action_signature = ""
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
            else: # hard, chaos, multi_agent_resolver, frontier
                if should_escalate:
                    action = Action(action_type="escalate",
                                    value=f"Escalating: sentiment {t.sentiment:.2f}, category {t.category.value}",
                                    reasoning="High severity — requires senior team")
                elif task_level == "frontier" and obs.governance_hint in ("block", "human_review_required") and "policy_reference" not in obs.evidence_collected:
                    action = Action(action_type="tool_call", value="policy_lookup", reasoning="Collect policy evidence")
                elif task_level == "frontier" and "fraud_risk" in obs.high_risk_flags and "fraud_check_id" not in obs.evidence_collected:
                    action = Action(action_type="tool_call", value="fraud_screen", reasoning="Collect fraud evidence")
                elif task_level == "frontier" and "account_takeover" in obs.high_risk_flags and "kyc_verified" not in obs.evidence_collected:
                    action = Action(action_type="tool_call", value="kyc_verify", reasoning="Collect KYC evidence")
                elif task_level == "frontier" and "pii_exposure" in obs.high_risk_flags and "pii_redaction_proof" not in obs.evidence_collected:
                    action = Action(action_type="tool_call", value="trust_safety_review", reasoning="Collect PII safety evidence")
                elif task_level == "frontier" and obs.governance_hint in ("human_review_required", "legal_hold"):
                    action = Action(action_type=obs.governance_hint, value="Policy gated handoff", reasoning="Governance gate")
                else:
                    action = Action(action_type="resolve",
                                    value=f"Apologize for the inconvenience. {kb_steps}. Following up to confirm.",
                                    reasoning="KB-aligned resolution with empathy")

        action_signature = f"{action.action_type}:{action.value}"
        if action_signature == previous_action_signature:
            repeated_action_count += 1
        else:
            repeated_action_count = 1
            previous_action_signature = action_signature

        # Demo anti-loop guard: prevent repeated tool-call farming.
        if (
            task_level == "frontier"
            and action.action_type == "tool_call"
            and repeated_action_count >= 3
        ):
            fallback_action = "legal_hold" if obs.governance_hint == "legal_hold" else "human_review_required"
            action = Action(
                action_type=fallback_action,
                value="Loop-prevention policy handoff",
                reasoning="Repeated identical tool_call detected in demo loop",
            )
            action_signature = f"{action.action_type}:{action.value}"
            repeated_action_count = 1
            previous_action_signature = action_signature

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

    # Include demo episodes in aggregate metrics so dashboard reflects usage.
    state = env.state()
    cumulative_reward = float(state.get("cumulative_reward", 0.0))
    _metrics["scores"].append(cumulative_reward)
    _metrics["tasks"].append(task_level)
    total_tickets = max(1, int(state.get("total_tickets", len(env.tickets) or 1)))
    tickets_handled = int(state.get("tickets_handled", 0))
    telemetry = state.get("telemetry", {})
    _episode_telemetry.append({
        "task_level": task_level,
        "cumulative_reward": cumulative_reward,
        "resolution_rate": float(tickets_handled / total_tickets),
        "escalation_rate": float(telemetry.get("safe_handoff", 0)) / total_tickets,
        "safe_handoff_rate": float(telemetry.get("safe_handoff", 0)) / total_tickets,
        "blocked_unsafe_action_rate": float(telemetry.get("unsafe_action_blocked", 0)) / total_tickets,
        "wrongful_autonomy_rate": float(telemetry.get("wrongful_autonomy", 0)) / total_tickets,
        "tool_calls_per_ticket": float(telemetry.get("tool_calls", 0)) / total_tickets,
        "tool_fallback_rate": float(telemetry.get("tool_fallbacks", 0)) / max(1, float(telemetry.get("tool_calls", 0))),
    })
    auto_title = f"Auto Demo Episode · {task_level} · {agent_type}"
    auto_log = _format_episode_log(
        task_level=task_level,
        source="demo",
        cumulative_reward=cumulative_reward,
        state=state,
    )
    _persist_run(auto_title, auto_log)

    return {
        "task_level": task_level,
        "agent_type": agent_type,
        "model_name": model_name,
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
        "task_distribution": {t: _metrics["tasks"].count(t) for t in set(_metrics["tasks"])},
        "scorecard": aggregate_slo_kpi(_episode_telemetry) if _episode_telemetry else {},
    }


@app.get("/scorecard")
async def scorecard():
    return aggregate_slo_kpi(_episode_telemetry)


@app.get("/export/scorecard")
async def export_scorecard():
    scorecard_payload = aggregate_slo_kpi(_episode_telemetry)
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "scorecard_report.json")
    with open(output_path, "w") as f:
        json.dump(scorecard_payload, f, indent=2)
    return {"status": "ok", "output_path": output_path, "scorecard": scorecard_payload}


@app.get("/providers/health")
async def providers_health(x_session_id: Optional[str] = Header(default=None)):
    env = _sessions.get(x_session_id or "")
    if env is None:
        raise HTTPException(400, "No active session.")
    return env.toolhub.providers_health()

def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting FrontierOps Arena on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
