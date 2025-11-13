# Reinforcement Learning: Customer Support Chatbot with Helpfulness Preferences

Design a preference-optimized customer support assistant that learns to balance helpfulness, safety, and brand voice using Tinker’s reinforcement learning (RLHF) tooling. This tutorial outlines the full workflow from data collection to evaluation so you can publish a standout community project.

## Project Overview

- **Goal**: Fine-tune a chat model that resolves support tickets while maximizing human-defined helpfulness preferences.
- **Approach**: Collect pairwise user judgments, train a reward model (optional), and run direct preference RL vs reward-model RL for comparison.
- **Why it shines**: Demonstrates alignment techniques, uses Tinker’s RL abstractions (`recipes/preference/`), and highlights nuanced evaluation with `evaluators.py` + custom rubrics.

## Learning Outcomes

- Structure preference datasets (`prompt`, `response_a`, `response_b`, `winner`) for RLHF.
- Leverage `tinker-cookbook` RL loops (`preference/train_reward_model.py`, `preference/direct_preference_rl.py`).
- Implement custom evaluation callbacks via `evaluators.py`, `types.py`, and InspectAI integrations.
- Analyze trade-offs between direct preference optimization and reward-model pipelines.

## Prerequisites

- Python 3.8+ and dependencies from `requirements.txt`.
- Tinker API key (`export TINKER_API_KEY=...`).
- Optional: Access to existing customer support transcripts or synthetic generator.
- Familiarity with RLHF concepts and evaluation metrics (win rate, regret, conversation quality).

### Install Extras

```bash
pip install -r requirements.txt
pip install datasets pandas jinja2 numpy scipy scikit-learn bert-score rouge-score
```

## Suggested Repository Additions

```
.
├── data/
│   ├── support_preference_train.jsonl
│   ├── support_preference_val.jsonl
│   └── support_prompts.jsonl
├── configs/
│   ├── rl_direct_preference.yaml
│   └── rl_reward_model.yaml
├── evaluators/
│   ├── support_helpfulness.py
│   └── safety_guardrails.py
└── tutorial_rl_customer_support_chatbot.md  # ← this file
```

## Step 1 — Generate or Collect Preference Data

1. **Seed prompts**: Use historic support questions or generate synthetic ones (billing issues, technical troubleshooting, account access).
2. **Draft candidate responses**:
   - Baseline: general-purpose model (`meta-llama/Llama-3.1-8B-Instruct`).
   - Variations: adjust tone, completeness, accuracy.
   - Scriptable via `generate_training_data.py` or a notebook.
3. **Collect pairwise labels**:
   - Annotators choose the more helpful response.
   - Capture rationales for qualitative analysis.
4. Store in JSONL:
   ```json
   {
     "prompt": "...",
     "response_a": "...",
     "response_b": "...",
     "winner": "a",
     "tags": {"intent": "billing_refund", "difficulty": "medium"}
   }
   ```

## Step 2 — Reward Model vs Direct Preference Setup

### Reward Model Pipeline

1. Use `recipes/preference/train_reward_model.py` with your config:
   ```bash
   python -m recipes.preference.train_reward_model \
     --config configs/rl_reward_model.yaml \
     --train-file data/support_preference_train.jsonl \
     --eval-file data/support_preference_val.jsonl
   ```
2. Config tips:
   - Base model: `meta-llama/Llama-3.1-8B-Instruct`.
   - LoRA rank 8–16.
   - Binary cross-entropy loss on preference logits.
3. Evaluate reward model accuracy on validation pairs and inspect calibration plots.

### Direct Preference RL Pipeline

1. Run `recipes/preference/direct_preference_rl.py`:
   ```bash
   python -m recipes.preference.direct_preference_rl \
     --config configs/rl_direct_preference.yaml \
     --prompt-file data/support_prompts.jsonl
   ```
2. Key hyperparameters:
   - `kl_beta`: 0.05–0.2 (controls divergence from base model).
   - `learning_rate`: 5e-6 to 1e-5.
   - `rollouts_per_prompt`: 4–8 for stable gradients.
3. Logging:
   - Enable `wandb` or local JSON logging.
   - Track preference win rates per mini-batch and KL divergence.

## Step 3 — Shared Utilities & Evaluation Hooks

- Implement `evaluators/support_helpfulness.py` to score generated responses with heuristics (response length, presence of apology, actionability).
- Use `evaluators/types.py` to register custom evaluators for the RL loop.
- Integrate InspectAI:
  ```bash
  python run_inspect_evals.py \
    --config configs/support_inspect.yaml \
    --input data/support_eval_prompts.jsonl
  ```
- Compare metrics:
  - Direct preference RL vs reward-model RL (table of win rates, mean reward, conversation satisfaction).
  - Baseline vs fine-tuned responses (A/B tests on held-out prompts).

## Step 4 — Offline and Online Evaluation

### Offline

- Use `evaluate_model.py` to script batch inference on `data/support_eval_prompts.jsonl`.
- Compute metrics: helpfulness win rate, response latency, refusal rate.
- Visualize with `matplotlib`/`seaborn` charts (loss curves, win rate progression).

### Online Simulation

- Build a simulated user loop:
  ```python
  from evaluators.support_helpfulness import simulate_user
  reward = simulate_user(prompt, model_response)
  ```
- Track cumulative reward and conversation success rate.

### Human Review

- Ask support agents to grade transcripts on clarity, completeness, empathy.
- Provide annotation interface (e.g., in Google Sheets or Label Studio).

## Step 5 — Comparative Analysis

Document differences between:

- **Direct preference RL**: Simpler pipeline, faster iteration, potentially higher variance.
- **Reward-model RL**: More stable but requires quality reward model training.
- Provide charts/tables showcasing trade-offs and sample transcripts demonstrating improvements.

## Launch Checklist

- [ ] Preference dataset cleaned, anonymized, and documented.
- [ ] RL configs checked into `configs/`.
- [ ] Training logs and sample outputs saved to `reports/`.
- [ ] Evaluation scripts produce reproducible metrics.
- [ ] README/notebook demonstrates inference & evaluation end-to-end.
- [ ] Ethical considerations noted (escalation policy, refusal behavior).

## Bonus Extensions

- **Safety Guardrails**: Incorporate refusal templates or policy classifiers using `evaluators/safety_guardrails.py`.
- **Tone Control**: Add LoRA adapters per customer segment (SMB vs enterprise) and switch at inference with adapter fusion.
- **Active Learning**: Stream new agent conversations into the preference dataset and retrain periodically.
- **Production Demo**: Create a Streamlit or Gradio app showing real-time preference scores and explanations.

Package these assets with a polished write-up to submit as a community project. Highlight before/after transcripts and preference win-rate gains to stand out.

