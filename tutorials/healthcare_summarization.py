"""
Domain-specific adaptation example: healthcare summarization with Tinker.

This starter uses a handful of anonymized clinical notes to demonstrate how to
fine-tune a large model with LoRA adapters. Replace the toy examples with a
properly de-identified corpus that complies with medical privacy regulations
before running at scale.

Prerequisites:
    pip install tinker
    export TINKER_API_KEY=<your-key>
"""

import tinker
from tinker import types


def process_example(example: dict, tokenizer) -> types.Datum:
    """
    Encode a medical note and its concise summary.

    The prompt is treated as context-only (weight 0) while the summary tokens
    drive the loss.
    """
    prompt = f"Note: {example['note']}\nSummary:"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)

    completion = f" {example['summary']}\n"
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
    )


def main() -> None:
    service_client = tinker.ServiceClient()
    base_model = "meta-llama/Llama-3.1-70B"
    training_client = service_client.create_lora_training_client(base_model=base_model)
    tokenizer = training_client.get_tokenizer()

    examples = [
        {
            "note": "Patient has fever and cough.",
            "summary": "Symptoms: Fever, cough. Diagnosis: Possible flu.",
        },
        {
            "note": "Patient reports headache and nausea with mild dehydration.",
            "summary": "Symptoms: Headache, nausea, dehydration. Plan: Rehydrate, monitor.",
        },
    ]

    processed_examples = [process_example(example, tokenizer) for example in examples]

    for _ in range(10):
        training_client.forward_backward(
            processed_examples, loss_identifier="cross_entropy"
        ).result()
        training_client.optim_step(types.AdamParams(learning_rate=2e-5)).result()
        print("Domain adaptation step complete")

    sampling_client = training_client.save_weights_and_get_sampling_client(
        name="health-summarizer"
    )
    prompt = types.ModelInput.from_ints(
        tokenizer.encode("Note: Patient reports headache and nausea.\nSummary:")
    )
    params = types.SamplingParams(max_tokens=30, temperature=0.4)
    result = sampling_client.sample(prompt=prompt, sampling_params=params).result()
    print(tokenizer.decode(result.sequences[0].tokens))


if __name__ == "__main__":
    main()
