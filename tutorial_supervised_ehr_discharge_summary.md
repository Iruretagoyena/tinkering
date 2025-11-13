# Supervised Learning: Structured EHR → Discharge Summary Generator

Build a clinician-facing discharge summary assistant by fine-tuning a Tinker model on structured electronic health record (EHR) data. This README walks you from dataset preparation through evaluation so you can ship a polished community tutorial or featured project submission.

## Project Overview

- **Goal**: Teach Tinker to translate structured clinical fields (labs, meds, vitals, diagnoses) into natural-language discharge summaries.
- **Audience**: Researchers and ML engineers interested in healthcare NLP, summarization, and responsible model evaluation.
- **Why it shines**: Demonstrates Tinker’s supervised pipelines, uses real-world data modalities, and supports rigorous benchmarking (ROUGE/BERTScore + human review).

## Learning Outcomes

- Loading tabular/JSON EHR datasets and packaging them for Tinker supervised fine-tuning.
- Extending the scientific code tutorial scripts (`generate_training_data.py`, `train_scientific_code_model.py`, `evaluate_model.py`) for seq2seq summarization.
- Integrating rubric-based scoring plus clinician-in-the-loop evaluation.
- Producing visual rollouts with `tinker-cookbook` utilities (`viz_sft_dataset.py`, `nll_evaluator.py`) to highlight improvements over baseline models.

## Prerequisites

- Python 3.8+ and the dependencies in `requirements.txt`
- Tinker API access (`export TINKER_API_KEY=...`)
- Optional: InspectAI keys or other evaluator credentials if benchmarking externally
- Familiarity with HIPAA-style de-identification or synthetic data sources

### Install & Environment

```bash
pip install -r requirements.txt
# Optional helpers for evaluation and reporting
pip install bert-score rouge-score matplotlib pandas seaborn
```

## Project Structure

Start from the existing tutorial layout:

```
.
├── generate_training_data.py
├── train_scientific_code_model.py
├── evaluate_model.py
├── data/
│   └── test_examples.json
└── tutorial_supervised_ehr_discharge_summary.md  # ← this file
```

You will add:

- `data/ehr_train.jsonl` – synthetic or de-identified EHR records
- `data/ehr_val.jsonl` – validation set with ground-truth summaries
- `configs/ehr_supervised.yaml` – hyperparameters, LoRA config, evaluation hooks

## Step 1 — Curate Structured EHR Data

1. Choose a compliant dataset: MIMIC-III, i2b2 (with PHI removed), or handcrafted synthetic data.
2. Normalize each encounter into a schema:
   ```json
   {
     "patient_id": "12345",
     "diagnoses": ["Congestive heart failure", "Hypertension"],
     "medications": [{"name": "Furosemide", "dose": "20 mg"}],
     "labs": [{"name": "BNP", "value": 720, "units": "pg/mL"}],
     "vitals": {"heart_rate": 95, "blood_pressure": "140/90"},
     "procedures": ["Echocardiogram"],
     "summary": "The patient presented with decompensated CHF..."
   }
   ```
3. Split into train/validation/test sets. Save as JSONL so each line is one example.
4. Optionally augment with rule-based draft summaries to boost coverage.

### Data Loader Hook

Clone `generate_training_data.py` into `generate_ehr_training_data.py` and adapt `generate_training_examples()` to parse JSONL inputs into the `{"input": ..., "output": ...}` format expected by Tinker’s supervised datasets.

```python
from pathlib import Path
import json

def load_ehr_examples(path: Path):
    for line in path.open():
        record = json.loads(line)
        yield {
            "prompt": format_ehr_prompt(record),
            "completion": record["summary"]
        }
```

Use `tinker-cookbook/recipes/sft/datasets.py` utilities if you prefer templated dataset builders.

## Step 2 — Format Prompts & Templates

Create a Jinja or f-string template that serializes structured fields into a readable prompt:

```
You are writing a hospital discharge summary.
Patient Diagnoses:
- Congestive heart failure
- Hypertension

Medications:
- Furosemide 20 mg PO daily

Key Labs:
- BNP: 720 pg/mL

Vitals:
- Heart Rate: 95 bpm
- Blood Pressure: 140/90 mmHg

Compose a concise discharge summary that covers hospital course, medications, and follow-up instructions.
```

Keep templates deterministic so ROUGE-based metrics remain meaningful. Store templates in `templates/ehr_prompt.txt` or inline within your loader.

## Step 3 — Configure Tinker Fine-Tuning

1. Use the existing `train_scientific_code_model.py` as a starting point.
2. Update:
   - Model choice to `meta-llama/Llama-3.1-8B-Instruct` (or healthcare-safe alternative).
   - LoRA ranks (e.g., rank 16, alpha 32) validated by `tinker-cookbook/recipes/lora/configs`.
   - Sequence lengths (prompt + summary ~ 2048 tokens).
   - Optimizer (AdamW with `lr=1e-4`, warmup steps ~500).
3. Configure logging:
   ```python
   trainer = TinkerTrainer(
       logging_steps=50,
       evaluation_strategy="steps",
       eval_steps=200,
       save_strategy="epoch",
       metrics=["rougeL", "bertscore", "loss"]
   )
   ```
4. Save hyperparameters into `configs/ehr_supervised.yaml` so others can reproduce the run.

### Cookbook Assist

Leverage `tinker-cookbook/recipes/sft/train.py` to avoid boilerplate. Pass your dataset builder and config file to the CLI:

```bash
python -m recipes.sft.train \
  --config configs/ehr_supervised.yaml \
  --train-file data/ehr_train.jsonl \
  --eval-file data/ehr_val.jsonl
```

## Step 4 — Evaluate & Benchmark

### Automatic Metrics

- Compute ROUGE-L, ROUGE-2, and BERTScore using `evaluate_model.py` or InspectAI (via `run_inspect_evals.py`).
- Track negative log-likelihood curves with `tinker-cookbook/tools/nll_evaluator.py`.
- Log results to `reports/ehr_metrics.csv`.

### Human-in-the-loop

- Recruit clinicians or medical writers to rate summaries on factuality, safety, and completeness.
- Provide an evaluation template (Likert scales with free-text feedback).
- Aggregate qualitative notes and quote key findings in your write-up.

### Visualization

- Use `viz_sft_dataset.py` to show prompt/summary distributions.
- Plot training vs evaluation loss, ROUGE progression, and hallucination rates.

## Step 5 — Deliverables & Write-Up

Create a polished tutorial package:

- `README.md` (or blog post) summarizing motivation, dataset, training setup, evaluation, and future work.
- Jupyter notebook or CLI script demonstrating inference on new patients.
- Comparison table: baseline vs fine-tuned metrics plus human scores.
- Safety considerations and guidance on deploying in clinical settings.

## Bonus Extensions

- **InspectAI Integration**: Configure `run_inspect_evals.py` for automated rubric-based clinician judgment.
- **Multi-task Learning**: Add note generation or medication reconciliation as auxiliary targets.
- **Structured Output Control**: Use Tinker function-calling APIs to guarantee section headings (History, Medications, Follow-up).
- **Privacy Story**: Highlight synthetic data generation and PHI scrubbing workflows to reassure reviewers.

## Submission Checklist

- [ ] Data provenance documented (source, license, de-identification).
- [ ] Training and evaluation scripts runnable end-to-end.
- [ ] Metrics visualized (plots/screenshots included).
- [ ] Human evaluation protocol described.
- [ ] Clear instructions to reproduce results on Tinker.

When you are ready, package your materials per the community guidelines and link to this README so others can follow along. Good luck building the next featured healthcare project!

