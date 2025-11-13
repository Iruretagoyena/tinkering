"""
Preference optimization / RLHF starter using Tinker.

This script sketches how to prepare prompt-response preference pairs where a
chosen answer should be preferred over a rejected alternative. The example
keeps the loss function abstract (`preference_loss`) so that you can plug in the
reward modeling utilities from the full Tinker cookbook.

Prerequisites:
    pip install tinker
    export TINKER_API_KEY=<your-key>
"""

import tinker
from tinker import types


def process_pair(pair: dict, tokenizer) -> dict:
    """
    Convert a preference pair into the data format expected by preference losses.

    The returned dictionary mirrors the structure consumed by the `preference_loss`
    helper in the Tinker cookbook: prompt context plus chosen/rejected completions.
    """
    prompt_tokens = tokenizer.encode(pair["prompt"], add_special_tokens=True)
    chosen_tokens = tokenizer.encode(pair["chosen"], add_special_tokens=False)
    rejected_tokens = tokenizer.encode(pair["rejected"], add_special_tokens=False)

    return dict(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        chosen=types.ModelInput.from_ints(chosen_tokens),
        rejected=types.ModelInput.from_ints(rejected_tokens),
    )


def main() -> None:
    service_client = tinker.ServiceClient()
    base_model = "meta-llama/Llama-3.1-8B"
    training_client = service_client.create_lora_training_client(base_model=base_model)
    tokenizer = training_client.get_tokenizer()

    pairs = [
        {
            "prompt": "Explain gravity.",
            "chosen": "Gravity is the force that attracts objects toward each other.",
            "rejected": "Gravity is magic.",
        }
    ]

    processed_pairs = [process_pair(pair, tokenizer) for pair in pairs]

    for _ in range(5):
        training_client.forward_backward(
            processed_pairs, loss_identifier="preference_loss"
        ).result()
        training_client.optim_step(types.AdamParams(learning_rate=1e-5)).result()
        print("Preference optimization step complete")

    sampling_client = training_client.save_weights_and_get_sampling_client(
        name="aligned-model"
    )
    prompt = types.ModelInput.from_ints(tokenizer.encode("What is AI?"))
    result = sampling_client.sample(
        prompt=prompt, sampling_params=types.SamplingParams(max_tokens=50)
    ).result()
    print(tokenizer.decode(result.sequences[0].tokens))


if __name__ == "__main__":
    main()
