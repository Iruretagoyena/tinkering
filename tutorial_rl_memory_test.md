# RL Memory Test: Evaluating Long-Horizon Recall in LLMs

Prototype an RL benchmark that tests an LLM’s ability to retain and recall information across extended interactions. This tutorial outlines how to design the environment, instrumentation, and analysis required to publish a community-ready research project.

## Project Overview

- **Goal**: Build a reinforcement learning environment where the agent must memorize and recall sequences (e.g., digit strings, instructions) over many turns.
- **Why it fits**: Aligns with the community’s suggested RL memory challenge, showcases experimental rigor, and spotlights Tinker’s RL loop flexibility.
- **Deliverables**: Environment code, baseline agents, evaluation metrics, and visualization assets.

## Learning Outcomes

- Implement a lightweight RL environment compatible with Tinker’s `rl_loop.py`.
- Generate synthetic curricula with controllable difficulty (sequence length, noise, distractors).
- Train policy gradients (PPO) or Q-learning agents using Tinker wrappers.
- Analyze sample efficiency, forgetting curves, and generalization.

## Prerequisites

- Python 3.8+, `requirements.txt` dependencies.
- Tinker API key.
- Familiarity with RL fundamentals (policy gradient, reward shaping).

### Additional Dependencies

```bash
pip install numpy gymnasium torch matplotlib seaborn pandas
```

## Directory Blueprint

```
.
├── memory_env/
│   ├── __init__.py
│   ├── sequence_env.py
│   └── curricula.py
├── configs/
│   └── rl_memory_test.yaml
├── scripts/
│   ├── train_memory_agent.py
│   └── evaluate_memory_agent.py
├── reports/
│   └── memory_results.csv
└── tutorial_rl_memory_test.md  # ← this file
```

## Step 1 — Environment Design

### Core Mechanics

1. At episode start, sample a random sequence (digits, words, JSON objects).
2. Present tokens over `N` steps with distractor messages.
3. After delay, prompt the agent to recall the sequence.
4. Reward structure:
   - +1 for each correct symbol in order.
   - Optional bonus for perfect recall; penalty for hallucinations.

### Implementation Sketch

```python
import numpy as np
from types import SimpleNamespace

class SequenceRecallEnv:
    def __init__(self, vocab_size=10, seq_len=6, delay=4, noise=0.2):
        self.config = SimpleNamespace(vocab_size=vocab_size,
                                      seq_len=seq_len,
                                      delay=delay,
                                      noise=noise)

    def reset(self):
        self.sequence = np.random.randint(0, self.config.vocab_size, self.config.seq_len)
        self.step_idx = 0
        return f"Memorize: {' '.join(map(str, self.sequence))}"

    def step(self, action):
        # action: model response string
        reward = compute_reward(action, self.sequence)
        done = True
        return "", reward, done, {"target": self.sequence}
```

Wrap the environment with Tinker’s RL interface inside `memory_env/sequence_env.py`.

## Step 2 — Curriculum & Difficulty Scaling

- `curricula.py` should yield environment configs with increasing `seq_len`, distractor density, and delay.
- Include curriculum schedules:
  - **Fixed**: constant difficulty.
  - **Progressive**: increase sequence length after consistent success.
  - **Adaptive**: adjust based on recent reward trends.

## Step 3 — Training Agents with Tinker

1. Use `tinker-cookbook/recipes/rl/rl_loop.py` as the backbone.
2. Configure PPO-style training in `configs/rl_memory_test.yaml`:
   - `learning_rate`: 1e-5
  - `kl_penalty`: 0.01
  - `rollouts_per_step`: 32
  - `max_prompt_length`: 1024 tokens
3. Baselines to compare:
   - **Supervised Imitation**: memorize via teacher forcing.
   - **PPO Agent**: learns through environment feedback.
   - **Memory-augmented**: supply scratchpad tokens or chain-of-thought hints.
4. Launch training:

```bash
python scripts/train_memory_agent.py \
  --config configs/rl_memory_test.yaml \
  --curriculum progressive
```

## Step 4 — Evaluation & Diagnostics

### Automated Checks

- Use `scripts/evaluate_memory_agent.py` to compute recall accuracy by sequence length.
- Log time-to-convergence, average reward, and KL divergence.
- Export metrics to `reports/memory_results.csv`.

### Qualitative Analysis

- Record trajectories and visualize attention to distractors.
- Inspect failure cases (e.g., off-by-one errors, truncation).
- Compare reasoning prompts between successful and failed attempts.

### Visualization Ideas

- **Success vs Sequence Length**: line chart showing recall accuracy over curriculum levels.
- **Learning Curves**: reward per training step.
- **Heatmaps**: token-level correctness.

## Step 5 — Report & Packaging

- Summarize experimental setup, hyperparameters, and results.
- Provide reproducibility instructions (seed management, dataset generation).
- Highlight key insights (e.g., memory decay after long delays).
- Offer a notebook that walks through inference and manual evaluation.

## Bonus Extensions

- **Multi-modal Memory**: Mix text with images (use placeholders or ASCII descriptions).
- **Noisy Feedback**: Inject stochastic rewards to test robustness.
- **Meta-RL**: Train agents that adapt to new sequence distributions quickly.
- **Human Benchmark**: Compare LLM performance to human recall baselines.

## Submission Checklist

- [ ] Environment code documented and packaged in `memory_env/`.
- [ ] Training/evaluation scripts runnable with configs.
- [ ] Metrics & plots saved under `reports/`.
- [ ] README/notebook explains results and limitations.
- [ ] Safety/ethical considerations addressed (e.g., hallucination risks).
- [ ] Future work suggestions for community collaboration.

Wrap up with a post highlighting memory retention curves and interesting failure cases to inspire community discussion.

