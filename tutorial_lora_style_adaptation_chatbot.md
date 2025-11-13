# LoRA Style Adaptation: Brand-Aware Chatbot Personality Tuning

Create a branded conversational assistant by layering lightweight LoRA adapters onto a Tinker base model. This tutorial guides you through dataset design, multi-style training, evaluation, and interactive demos tailored for community showcase submissions.

## Project Overview

- **Goal**: Build a chatbot that can switch among multiple brand personas (formal, playful, luxury, technical) using LoRA adapters.
- **Why it fits**: Highlights Tinker’s strength for rapid product prototyping, delivers a visual demo, and showcases adapter fusion + preference feedback.
- **Outcome**: Reusable LoRA weights, persona-switching inference script, and documentation for a polished tutorial.

## Learning Outcomes

- Craft conversation datasets with stylistic annotations.
- Train LoRA adapters using `train.py`, `renderers.py`, and cookbook utilities.
- Implement runtime adapter selection/fusion inside the sampling client.
- Evaluate style adherence with automatic metrics and human preference loops.

## Prerequisites

- Python 3.8+, `requirements.txt` dependencies.
- Tinker API credentials.
- Optional: brand guidelines or copywriting style guides.

### Suggested Extras

```bash
pip install pandas numpy jinja2 evaluate gradio matplotlib seaborn bert-score
```

## Repo Additions

```
.
├── data/
│   ├── persona_training.jsonl
│   ├── persona_validation.jsonl
│   └── persona_prompts.jsonl
├── configs/
│   └── lora_persona.yaml
├── scripts/
│   └── sample_persona_chat.py
├── notebooks/
│   └── persona_demo.ipynb
└── tutorial_lora_style_adaptation_chatbot.md  # ← this file
```

## Step 1 — Design Persona Dataset

1. Select 3–4 personas (e.g., **Formal Concierge**, **Playful Sidekick**, **Technical Advisor**).
2. Draft prompts covering customer journeys: onboarding, troubleshooting, upsells.
3. Author reference responses matching each persona, or bootstrap using prompt engineering.
4. Store in JSONL:
   ```json
   {
     "prompt": "How can I reset my password?",
     "persona": "Playful Sidekick",
     "response": "No worries! Let's zap that password back to life..."
   }
   ```
5. Optionally add metadata (tone sliders, compliance tags) for evaluation.

### Dataset Loader

- Extend `generate_training_data.py` into `generate_persona_training_data.py`.
- Map persona label to specific system prompt templates (see Step 2).

## Step 2 — Persona Prompt Templates

Define persona-specific instructions stored in `templates/personas/`:

```
{{persona}} Persona Guide
- Tone: Friendly, upbeat, playful metaphors
- Constraints: Always provide actionable steps
Conversation:
{{user_prompt}}
```

Use `renderers.py` from the cookbook to combine templates with conversation history. Keep prompts deterministic so evaluation comparisons remain fair.

## Step 3 — Train LoRA Adapters

1. Duplicate `train_scientific_code_model.py` to `train_persona_chatbot.py`.
2. Adjust:
   - Base model: `meta-llama/Llama-3.1-8B-Instruct` or similar.
   - LoRA rank 8, alpha 16, dropout 0.05.
   - Batch size 64 tokens, gradient accumulation as needed.
3. Use persona-specific adapters:
   - Train one adapter per persona (multi-run) or a single adapter conditioned on persona embeddings.
4. Example command:
   ```bash
   python train_persona_chatbot.py \
     --config configs/lora_persona.yaml \
     --persona "Playful Sidekick"
   ```
5. Save adapter weights under `outputs/lora/persona_name/`.

### Adapter Fusion (Optional)

- Use `tinker-cookbook/tools/adapter_fusion.py` to blend personas (e.g., 70% Formal + 30% Playful).

## Step 4 — Inference & Demo

Implement `scripts/sample_persona_chat.py`:

```python
from tinker import SamplingClient
from adapters import load_persona_adapter

client = SamplingClient(model="meta-llama/Llama-3.1-8B-Instruct")
adapter = load_persona_adapter("outputs/lora/playful")
client.load_lora(adapter)
response = client.chat(persona_prompt, user_message)
```

### Interactive Demo

- Build a Gradio/Streamlit app with persona dropdown selection.
- Show side-by-side outputs (base vs persona).
- Provide slider to blend adapters in real time.

## Step 5 — Evaluation

### Automatic Metrics

- **Style Similarity**: Compare generated responses to reference persona outputs via embedding cosine similarity.
- **Formality Score**: Use readability metrics (e.g., Flesch-Kincaid).
- **Toxicity/Safety**: Integrate moderation API or InspectAI rubric.

### Preference Testing

- Create small paired datasets where evaluators choose the best persona match.
- Use `evaluators.py` to log win rates and failure cases.

### Visualization

- Plot persona adherence scores per prompt.
- Show radar charts of tone attributes for each adapter.
- Include samples illustrating style transformations.

## Bonus Extensions

- **Multi-persona Switching**: Condition on runtime metadata (user tier, locale) to auto-select adapters.
- **Preference Loop**: Feed user thumbs-up/down signals back into a lightweight direct preference RL step.
- **Content Filters**: Add guardrail adapters that can be fused with persona adapters.
- **Analytics Dashboard**: Use `viz_sft_dataset.py` to present persona coverage and conversation lengths.

## Submission Checklist

- [ ] Persona dataset documented and ethically sourced.
- [ ] LoRA training configs versioned in `configs/`.
- [ ] Adapters packaged (`.safetensors` + metadata).
- [ ] Demo app instructions provided.
- [ ] Evaluation results (tables/plots) included in README or notebook.
- [ ] Future work + community collaboration ideas listed.

Publish your tutorial with screenshots/GIFs of persona switching to maximize impact with the community reviewers.

