# Judge Scorecard (One-Page Evidence)

This page is the fastest way to evaluate the project across capability, safety, and reliability.

## Thesis
**AI Support Envoy is an evidence-gated RL support-ops benchmark that optimizes customer outcomes while constraining unsafe autonomy under high-risk and degraded-tool conditions.**

## 1) Performance Delta (Deterministic Offline Repro Pack)

Source: `results/final_summary_stats.json`

| Task | Baseline Mean ± Std | Trained Mean ± Std | Delta |
|:---|:---:|:---:|:---:|
| Easy | `0.3144 ± 0.0885` | `0.9811 ± 0.0550` | `+212.01%` |
| Medium | `-0.9544 ± 0.2926` | `0.6044 ± 0.2464` | `+163.33%` |
| Hard | `-0.7167 ± 0.2596` | `-0.1331 ± 0.3271` | `+81.43%` |
| Frontier | `-0.6156 ± 0.0698` | `-0.1307 ± 0.0576` | `+78.77%` |

## 2) Safety + Governance Evidence

Sources: `results/final_baseline_vs_trained.md`, `results/ablation_hack_penalty.json`

- Anti-reward-hacking: removing urgent-spam penalty improves spam policy reward by `+0.92`
- Governance ablation:
  - Unsafe always-resolve policy: `unsafe_wrongful_autonomy_count = 6`
  - Governance-compliant policy: `safe_wrongful_autonomy_count = 0`
- Frontier safety scorecard (trained/offline):
  - `Safe handoff rate = 0.667`
  - `Blocked unsafe action rate = 0.000`
  - `Wrongful autonomy rate = 0.667` (visible residual risk to improve)

## 3) Reliability + Ops Realism

Source: `results/scorecard_report.json`

- Provider-style mock APIs simulate latency, timeout, rate-limit, and fallback behavior.
- Tool reliability telemetry:
  - `tool_fallback_rate = 0.0667`
- Business/ops KPI rollup:
  - `containment_rate = 0.7333`
  - `avg_tool_calls_per_ticket = 0.2667`

## 4) Why This Differs From A Standard Support Bot

- Environment-level world model with queue/SLA/sentiment/governance state.
- RL reward shaping + ablation-backed safeguards, not prompt-only behavior.
- Evidence-gated action policy (`tool_call` before sensitive autonomy).
- Explicit safe fallback actions: `human_review_required`, `legal_hold`.

## 5) Repro Run Path (3 Commands)

```bash
python evaluate_models.py --offline --tasks easy,medium,hard,frontier --episodes 3 --seeds 41,42,43 --output results/final_baseline_vs_trained.md --trained-model offline_stub
python ablation_eval.py
python -m server.app   # then GET /export/scorecard (or generate scorecard_report.json via provided script path)
```

