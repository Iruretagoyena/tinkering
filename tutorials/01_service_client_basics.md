# Getting Started with the Tinker Service Client

Learn how to authenticate with Tinker, explore available base models, and run your first inference request. Complete this tutorial before diving into fine-tuning or evaluation.

---

## Prerequisites

- Python 3.8+
- `pip install -r requirements.txt`
- Export a valid `TINKER_API_KEY`

```bash
export TINKER_API_KEY="your-api-key"
```

If you prefer a `.env` file, follow the instructions in the project `QUICKSTART.md`.

---

## 1. Initialize the Service Client

The service client is your gateway to Tinker. It handles authentication and exposes helpers for listing capabilities, creating fine-tuning clients, and sampling from models.

```python
import os
import tinker

if "TINKER_API_KEY" not in os.environ:
    raise RuntimeError("Set TINKER_API_KEY before continuing")

service_client = tinker.ServiceClient()
print("Service client ready!")
```

### Troubleshooting
- `401` errors: double-check your API key value and confirm it has not expired.
- SSL or networking errors: ensure you can reach the Tinker API endpoints from your environment (corporate VPNs sometimes block traffic).

---

## 2. Discover Supported Models

Use `get_server_capabilities()` to see which base models are currently available for fine-tuning or direct sampling.

```python
capabilities = service_client.get_server_capabilities()

print("Supported models:")
for item in capabilities.supported_models:
    print(f"- {item.model_name} (context window: {item.context_window})")
```

Keep an eye on the context window and whether a model supports training (`item.can_train`) or only inference.

---

## 3. Run a Zero-Shot Completion

For quick experiments, request a sampling client directly from the service client. This lets you prompt a base model without fine-tuning.

```python
from tinker import types

sampling_client = service_client.create_sampling_client(
    model="Qwen/Qwen3-30B-A3B-Base"
)

prompt = "Task: Explain the concept of Fourier transform in two sentences.\n\nAnswer:"
prompt_tokens = sampling_client.get_tokenizer().encode(
    prompt,
    add_special_tokens=True
)
model_input = types.ModelInput.from_ints(prompt_tokens)

params = types.SamplingParams(
    max_tokens=200,
    temperature=0.4,
    stop=["\n\n"]
)

result = sampling_client.sample(
    prompt=model_input,
    sampling_params=params,
    num_samples=1
).result()

print(result.sequences[0].text)
```

### Tips
- Adjust `temperature` for more or less diverse outputs.
- Provide domain-specific instructions in the prompt to steer generations.
- Use `num_samples > 1` when you want multiple candidates to choose from.

---

## 4. Next Steps

- **Fine-tune with LoRA** using the [LoRA Fine-Tuning Workflow](02_lora_fine_tuning.md).
- **Automate experimenting** by wrapping the sampling call in helper functions.
- **Log outputs**: Tinker clients return token IDs; decode and store text outputs for reproducibility.

Once you are comfortable with these basics, you're ready to customize models for your scientific code generation tasks.
