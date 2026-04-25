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

### 📊 Results
Our reproducible evaluation (via `evaluate_models.py`) shows a transformative shift in agent capability:
*   **Easy (Triage):** +56% improvement.
*   **Medium (Prioritization):** **+1,872% improvement** (Learning to ignore label-bait).
*   **Hard (Resolution):** **+284% improvement** (Mastering multi-step KB navigation).

### 🔗 Reproducibility
The training process is fully documented in `train_colab.ipynb` and is 100% reproducible with fixed seeds and a verified OpenEnv reward model.

---
*Built for the Meta PyTorch OpenEnv Hackathon 2026 Grand Finale.*
