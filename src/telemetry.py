"""
Telemetry helpers for SLO/KPI aggregation and exports.
"""

from typing import Dict, List


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def aggregate_slo_kpi(episodes: List[Dict]) -> Dict:
    rewards = [float(e.get("cumulative_reward", 0.0)) for e in episodes]
    resolution_rates = [float(e.get("resolution_rate", 0.0)) for e in episodes]
    escalation_rates = [float(e.get("escalation_rate", 0.0)) for e in episodes]
    safe_handoff_rates = [float(e.get("safe_handoff_rate", 0.0)) for e in episodes]
    blocked_unsafe_rates = [float(e.get("blocked_unsafe_action_rate", 0.0)) for e in episodes]
    wrongful_autonomy_rates = [float(e.get("wrongful_autonomy_rate", 0.0)) for e in episodes]
    tool_calls_per_ticket = [float(e.get("tool_calls_per_ticket", 0.0)) for e in episodes]
    tool_fallback_rate = [float(e.get("tool_fallback_rate", 0.0)) for e in episodes]

    return {
        "episodes": len(episodes),
        "slo": {
            "avg_cumulative_reward": round(_safe_mean(rewards), 4),
            "p95_cumulative_reward": round(sorted(rewards)[int(0.95 * (len(rewards) - 1))], 4) if rewards else 0.0,
            "avg_resolution_rate": round(_safe_mean(resolution_rates), 4),
            "avg_escalation_rate": round(_safe_mean(escalation_rates), 4),
        },
        "safety": {
            "safe_handoff_rate": round(_safe_mean(safe_handoff_rates), 4),
            "blocked_unsafe_action_rate": round(_safe_mean(blocked_unsafe_rates), 4),
            "wrongful_autonomy_rate": round(_safe_mean(wrongful_autonomy_rates), 4),
            "tool_fallback_rate": round(_safe_mean(tool_fallback_rate), 4),
        },
        "business_kpi": {
            "containment_rate": round(max(0.0, 1.0 - _safe_mean(escalation_rates)), 4),
            "automation_confidence_index": round(max(0.0, _safe_mean(safe_handoff_rates) - _safe_mean(wrongful_autonomy_rates)), 4),
            "avg_tool_calls_per_ticket": round(_safe_mean(tool_calls_per_ticket), 4),
        },
    }

