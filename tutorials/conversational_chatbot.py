"""
Conversational AI / Chatbot fine-tuning example using Tinker.

This starter script demonstrates how to fine-tune a chat model with LoRA using a
small set of domain-specific customer support dialogues. It mirrors the
high-level workflow covered in the Tinker cookbook:
  1. Instantiate a `ServiceClient` and create a LoRA training client.
  2. Tokenize supervised chat examples with prompt vs. completion weights.
  3. Run a short training loop.
  4. Export the adapter weights and sample from the tuned model.

Prerequisites:
    pip install tinker
    export TINKER_API_KEY=<your-key>
"""

import tinker
from tinker import types


def process_example(example: dict, tokenizer) -> types.Datum:
    """
    Convert a chat example into a weighted Datum for supervised fine-tuning.

    The prompt tokens are masked (weight 0) while the assistant completion
    tokens contribute to the loss (weight 1). The shift between `input_tokens`
    and `target_tokens` is handled here to align with next-token prediction.
    """
    prompt = f"User: {example['user']}\nAssistant:"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)

    completion = f" {example['assistant']}\n\n"
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
    base_model = "meta-llama/Llama-3.1-8B"
    training_client = service_client.create_lora_training_client(base_model=base_model)
    tokenizer = training_client.get_tokenizer()

    examples = [
        {
            "user": "My order is late.",
            "assistant": "I'm sorry for the delay. Can you provide your order number?",
        },
        {
            "user": "How do I return an item?",
            "assistant": "You can return items within 30 days. Visit our returns page.",
        },
    ]

    processed_examples = [process_example(example, tokenizer) for example in examples]

    for _ in range(5):
        fwdbwd_future = training_client.forward_backward(
            processed_examples, loss_identifier="cross_entropy"
        )
        fwdbwd_future.result()
        training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
        print("Training step complete")

    sampling_client = training_client.save_weights_and_get_sampling_client(name="chatbot")
    prompt = types.ModelInput.from_ints(
        tokenizer.encode("User: Cancel my subscription.\nAssistant:")
    )
    params = types.SamplingParams(max_tokens=50, temperature=0.7)
    result_future = sampling_client.sample(prompt=prompt, sampling_params=params)
    result = result_future.result()
    print(tokenizer.decode(result.sequences[0].tokens))


if __name__ == "__main__":
    main()
