# AI Support Envoy: Mastering Customer Operations with GRPO
**🎥 Watch the Demo:** [YouTube Video Link](https://youtu.be/BQKfDsODfFk)

### 🚀 The Mission
Customer support is often the bottleneck of scaling businesses. Static RAG-based bots fail to take actions, while rule-based systems are too rigid. The **AI Support Envoy** is a next-generation agent trained in a high-fidelity **OpenEnv** environment using **Group Relative Policy Optimization (GRPO)** to master the three pillars of support: Triage, Prioritization, and Resolution.

### 🧠 Technical Innovation: Curriculum Learning
We didn't just train a model; we designed a curriculum.
1. **Phase 1 (Easy):** Mastering **Triage**. The agent learns the nuance between technical bugs and billing disputes.
2. **Phase 2 (Medium):** Mastering **Prioritization**. Training the model to identify VIP customers and urgent SLAs without being "label-baited."
3. **Phase 3 (Hard):** Mastering **Resolution**. Teaching the agent to navigate a Knowledge Base to solve complex, multi-step issues.

### 🔬 The Secret Sauce: GRPO & Unsloth
By leveraging **Unsloth** for 2x faster fine-tuning and **GRPO** for reasoning-aware policy updates, we achieved:
- **Massive Reward Trajectory:** Moving from a failing baseline of `-0.21` on complex tasks to a trained peak of **+0.65**.
- **Reasoning Chains:** The agent doesn't just output an action; it outputs a `reasoning` field that explains its "World Model" of the ticket.
- **Anti-Reward Hacking:** Our reward function was specifically patched to penalize "priority spamming," ensuring the agent learns genuine business value.

### Results
Our full-run evidence (`baseline_vs_trained_colab.json` + reward curves) shows a transformative shift in agent capability:
*   **Easy (Triage):** +56% improvement.
*   **Medium (Prioritization):** **+1,872% improvement** (Learning to ignore label-bait).
*   **Hard (Resolution):** **+284% improvement** (Mastering multi-step KB navigation).

### Solo Budget Repro Pack (Apr 25)
For constrained reruns, we also provide a deterministic multi-seed offline benchmark:
- Easy: **+212.0%** (`0.314 ± 0.089` → `0.981 ± 0.055`)
- Medium: **+163.3%** (`-0.954 ± 0.293` → `0.604 ± 0.246`)
- Hard: **+81.4%** (`-0.717 ± 0.260` → `-0.133 ± 0.327`)
- Frontier: **+78.8%** (`-0.616 ± 0.070` → `-0.131 ± 0.058`)

Anti-hacking ablation (`ablation_hack_penalty.json`) shows removing the urgent-spam penalty increases spam policy reward by **+0.92**, validating the defense against reward hacking.

### Frontier Upgrade: Near-Complete Automation Path
To move beyond standard support bots, we added a frontier mode that combines:
- multilingual voice/text normalization with code-mix handling
- multi-industry world modeling (ecommerce, telecom, healthcare/insurance, travel)
- six high-risk classes (PII, fraud, account takeover, prompt injection, legal threat, medical safety)
- evidence-gated autonomy via tool calls (`policy_lookup`, `fraud_screen`, `kyc_verify`, `trust_safety_review`)
- provider-style mock APIs with transient failures (rate-limit/timeout), latency, and deterministic fallbacks

This converts the environment from "just resolve tickets" into "resolve only when policy and evidence permit", with explicit fallback actions:
- `human_review_required`
- `legal_hold`

### Governance and Safety Evidence
Latest ablation bundle now includes governance behavior:
- unsafe always-resolve policy: `unsafe_wrongful_autonomy_count = 6`
- governance-compliant policy: `safe_wrongful_autonomy_count = 0`

Even when reward deltas are close on tiny offline runs, wrongful-autonomy elimination is the core proof that safeguards are active.

### Production SLO / KPI Scorecard
Server now exposes:
- `GET /scorecard` for live SLO + safety + business KPI rollups
- `GET /export/scorecard` to persist `results/scorecard_report.json`

Scorecard dimensions:
- safety: safe handoff rate, blocked unsafe action rate, wrongful autonomy rate
- safety/reliability: tool fallback rate from simulated provider degradation
- business: containment rate, automation confidence index, tool-calls-per-ticket

### Judge Review Fast Path
To reduce manual review friction, we provide:
- `results/judge_scorecard.md`: one-page performance/safety/reliability evidence summary
- `JUDGE_QA.md`: concise answers for architecture, safety, and reproducibility questions

### Reproducibility
The training process is documented in `train_colab.ipynb` and backed by:
- full-run evidence artifacts
- deterministic fallback reproducibility pack (`results/final_*`)
- explicit anti-hacking ablation evidence

---
*Built for the Meta PyTorch OpenEnv Hackathon 2026 Grand Finale.*
