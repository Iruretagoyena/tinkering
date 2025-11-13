"""
Tool-use and retrieval-augmented generation (RAG) starter using Tinker.

This script fine-tunes a model to emit structured tool-call annotations before
producing the final response. With larger datasets you can train the model to
select among multiple tools and hand off control to downstream orchestrators.

Prerequisites:
    pip install tinker
    export TINKER_API_KEY=<your-key>
"""

import tinker
from tinker import types


def process_example(example: dict, tokenizer) -> types.Datum:
    """
    Encode a single tool-use demonstration as a sequence-to-sequence target.

    The template encourages a structured format:
        Query (...)
        Tool (...)
        Response (...)
    """
    prompt = f"Query: {example['query']}\nTool:"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)

    completion = f" {example['tool_call']}\nResponse: {example['response']}\n"
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
    base_model = "Qwen/Qwen2-7B-Instruct"
    training_client = service_client.create_lora_training_client(base_model=base_model)
    tokenizer = training_client.get_tokenizer()

    examples = [
        {
            "query": "Weather in NYC?",
            "tool_call": "[call_weather_api: New York]",
            "response": "Sunny, 75F.",
        },
        {
            "query": "Latest headline about Mars missions.",
            "tool_call": "[search_news: Mars Exploration]",
            "response": "NASA announces new rover prototype for 2030 launch.",
        },
    ]

    processed_examples = [process_example(example, tokenizer) for example in examples]

    for _ in range(5):
        training_client.forward_backward(
            processed_examples, loss_identifier="cross_entropy"
        ).result()
        training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
        print("Tool-use training step complete")

    sampling_client = training_client.save_weights_and_get_sampling_client(name="rag-model")
    prompt = types.ModelInput.from_ints(
        tokenizer.encode("Query: Stock price of AAPL?\nTool:")
    )
    params = types.SamplingParams(max_tokens=20, temperature=0.2)
    result = sampling_client.sample(prompt=prompt, sampling_params=params).result()
    print(tokenizer.decode(result.sequences[0].tokens))


if __name__ == "__main__":
    main()
