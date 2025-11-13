"""
Math and logical reasoning enhancement example using Tinker's RL-ready workflow.

The goal is to encourage the model to produce structured, step-by-step solutions
for simple arithmetic prompts. This starter demonstrates how to weight solution
tokens more heavily so that the optimizer reinforces correct reasoning steps.
For a production RLHF setup, plug this data processing logic into the reward
pipeline outlined in the official cookbook.

Prerequisites:
    pip install tinker
    export TINKER_API_KEY=<your-key>
"""

import tinker
from tinker import types


def process_example(example: dict, tokenizer, reward_scale: float = 2.0) -> types.Datum:
    """
    Encode a math problem and its reasoning chain.

    We bias the loss toward the solution tokens by assigning them a larger
    weight (reward_scale). This keeps the starter compatible with
    `cross_entropy` while mimicking reward shaping.
    """
    prompt = f"Problem: {example['problem']}\nSolution:"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0.0] * len(prompt_tokens)

    completion = f" {example['solution']}\n"
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    completion_weights = [reward_scale] * len(completion_tokens)

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
    base_model = "Qwen/Qwen2-7B"
    training_client = service_client.create_lora_training_client(base_model=base_model)
    tokenizer = training_client.get_tokenizer()

    examples = [
        {"problem": "What is 2 + 2?", "solution": "Step 1: Add the numbers. 2 + 2 = 4."},
        {"problem": "Solve 3 * 4.", "solution": "Step 1: Multiply. 3 * 4 = 12."},
        {"problem": "Compute 10 - 6.", "solution": "Step 1: Subtract. 10 - 6 = 4."},
    ]

    processed_examples = [process_example(example, tokenizer) for example in examples]

    for _ in range(10):
        training_client.forward_backward(
            processed_examples, loss_identifier="cross_entropy"
        ).result()
        training_client.optim_step(types.AdamParams(learning_rate=5e-5)).result()
        print("RL-style step complete")

    sampling_client = training_client.save_weights_and_get_sampling_client(name="math-solver")
    prompt = types.ModelInput.from_ints(tokenizer.encode("Problem: What is 5 - 3?\nSolution:"))
    params = types.SamplingParams(max_tokens=30, temperature=0.0)
    result = sampling_client.sample(prompt=prompt, sampling_params=params).result()
    print(tokenizer.decode(result.sequences[0].tokens))


if __name__ == "__main__":
    main()
