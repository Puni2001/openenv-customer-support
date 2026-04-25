# Judge Q&A Cheat Sheet

## 1) How is this different from a normal RAG support bot?

RAG answers questions; this project trains a policy in a **stateful environment** with SLA deadlines, queue pressure, customer history, and governance constraints. The agent is scored on actions and outcomes, not just text quality.

## 2) How do you prevent unsafe autonomy?

- Six high-risk classes are detected (PII, fraud, account takeover, prompt injection, legal threat, medical safety).
- Sensitive cases are **evidence-gated** via `tool_call`.
- Unsafe flows are redirected to `human_review_required` or `legal_hold`.
- Wrongful autonomy is measured explicitly in telemetry and ablations.

## 3) What happens when tool APIs fail?

We simulate provider degradation (timeout/rate-limit), then route to deterministic fallback payloads. Reliability is tracked via `tool_fallback_rate`, and provider health is exposed through `/providers/health`.

## 4) What proves the reward function is not being gamed?

`results/ablation_hack_penalty.json` shows removing anti-spam penalty increases spam-policy reward (`+0.92`), proving the safeguard closes a real loophole.

## 5) Is this claiming full human replacement now?

No. The project demonstrates a **credible path to near-complete automation** with measurable safeguards and explicit human handoff gates for high-risk scenarios.

## 6) Why should this rank high among many submissions?

Because it combines:
- RL + OpenEnv world modeling,
- safety-governed autonomy,
- reproducible multi-seed evaluations,
- ablation-backed claims,
- reliability modeling under degraded APIs.

## 7) What is the quickest way for a judge to verify?

Run:
1. `evaluate_models.py` offline multi-seed benchmark
2. `ablation_eval.py`
3. Review `results/judge_scorecard.md` and `results/scorecard_report.json`

