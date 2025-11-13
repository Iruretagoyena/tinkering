# Evaluation and Monitoring

Use this guide to extend `evaluate_model.py`, capture meaningful metrics, and keep tabs on model quality after every training run.

---

## 1. Understand the Evaluation Script

The shipped script outlines an evaluation flow without assuming how you load adapters back from Tinker. Key functions:

- `load_test_data` reads `data/test_examples.json`
- `generate_code` prompts the sampling client and returns decoded text
- `check_syntax`, `check_imports`, and `check_keywords` compute lightweight quality signals
- `run_code_safely` executes snippets inside a temporary file with a timeout

Walk through the script once before you start modifying it.

---

## 2. Loading Your Saved Model

After fine-tuning, you typically receive adapter identifiers from the training client. To evaluate:

1. Recreate the service client
2. Use the identifier to obtain a sampling client (API specifics may differ per deployment)
3. Pass that client plus the tokenizer into `evaluate_example`

When running locally, you can skip API calls entirely and reuse the `sampling_client` returned by `train_model`.

---

## 3. Capturing Metrics Across the Suite

The pseudo-code block near the bottom of the script shows how to aggregate metrics. Expand it with a helper:

```python
def summarize(evaluations):
    total = len(evaluations)
    syntax_rate = sum(e["syntax_valid"] for e in evaluations) / total
    keyword_rate = sum(e["keyword_score"] for e in evaluations) / total
    execution_rate = sum(e["code_runs"] for e in evaluations) / total
    avg_length = sum(e["code_length"] for e in evaluations) / total
    return {
        "syntax_rate": syntax_rate,
        "keyword_rate": keyword_rate,
        "execution_rate": execution_rate,
        "avg_length": avg_length,
    }
```

Persist the resulting dictionary to disk as JSON so you can compare runs over time.

---

## 4. Extending Test Cases

`data/test_examples.json` uses the schema:

```json
{
  "task": "Calculate the mean and standard deviation of a list of numbers",
  "expected_keywords": ["mean", "std", "numpy", "np."]
}
```

Guidelines for authoring new tests:

- Choose tasks that stress different library imports and code structures
- Add at least one keyword per important API call you expect to see
- Keep the task description short so it resembles real prompts
- Mix easy and hard tasks to detect regression patterns

---

## 5. Monitoring in Production

Once you deploy your adapter:

- Log every prompt and generation (with PII-safe redaction) for later review
- Sample batches of generations daily and run them through `evaluate_example`
- Alert when syntax or execution success rates drop below agreed thresholds
- Version control your adapters and test sets to make rollback painless

---

## 6. Automating the Pipeline

Wrap evaluation inside a job runner or CI pipeline so it executes after every fine-tuning run:

```bash
python evaluate_model.py --model scientific-code-generator --verbose
```

Collect artifacts:

- Raw generations (store in a timestamped folder)
- Structured metrics JSON
- Execution logs or stderr traces

Feed those artifacts back into your experimentation tracker.

---

## 7. Next Steps

- Build a dashboard with your favorite BI tool to visualize evaluation metrics.
- Expand evaluation with domain-specific checkers (for example, static analysis or unit tests).
- Contribute back improved helpers to `evaluate_model.py` so future runs benefit from your instrumentation.
