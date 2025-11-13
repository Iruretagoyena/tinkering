# Data Preparation Patterns

Structured, diverse training data is the foundation of a reliable scientific code assistant. Use this tutorial to understand and extend `generate_training_data.py`.

---

## 1. Dataset Schema

Every example is a dictionary with two keys:

- `task`: Natural language description of the goal
- `code`: Reference Python solution

The script returns a list of these pairs and saves them to `data/training_examples.json`.

---

## 2. Coverage Areas

The default generator spans several scientific computing domains:

- Data visualization (Matplotlib, Seaborn)
- Statistical analysis (SciPy stats)
- Numerical methods (integration, optimization)
- Data processing (Pandas, scikit-learn)
- Signal processing (SciPy signal)

When adding new domains, keep the balance between libraries so the model generalizes instead of memorizing syntax for just one library.

---

## 3. Authoring High-Quality Examples

Follow these guidelines when crafting new examples:

1. **Single focused task**: Avoid combining unrelated operations in the same prompt.
2. **Executable code**: Ensure the snippet runs if the required variables are provided.
3. **Consistent style**: Use clear variable names and include comments where clarity is needed.
4. **Edge cases**: Include prompts that require parameter tuning, error handling, or custom plotting.
5. **Variety**: Mix simple and advanced tasks to keep the dataset interesting.

---

## 4. Generating the JSON File

Call the helper functions to produce the latest dataset:

```bash
python generate_training_data.py
```

This saves the file under `data/` and prints a coverage summary. Commit the generated JSON if you want collaborators to reuse the exact dataset.

---

## 5. Keeping Examples Maintainable

As the dataset grows, consider:

- Splitting generator logic into thematic helper functions
- Validating code snippets with `compile` or automated tests
- Documenting each example inline with a short comment
- Tracking dataset versions (for example, `data/training_examples_v2.json`)

If two examples become too similar, merge or delete one to keep the corpus lean.

---

## 6. Creating Benchmarks from the Dataset

Reuse the same task prompts in your evaluation suite to measure how well the model reproduces expected keywords or code structure. Store a subset as held-out validation data so you can monitor overfitting.

---

## 7. Next Steps

- Convert portions of the dataset into unit tests for generated code.
- Build synthetic examples programmatically (for example, with parameter sweeps).
- Share your curated datasets with the team through pull requests in this folder.
