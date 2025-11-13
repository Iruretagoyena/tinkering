"""
Train a scientific code generation model using Tinker.

This script demonstrates how to:
1. Set up a Tinker training client
2. Prepare training data in the correct format
3. Train a LoRA model
4. Save and sample from the trained model
"""

import json
import os
import numpy as np
from typing import List, Dict
import tinker
from tinker import types

def load_training_data(filename: str = "data/training_examples.json") -> List[Dict[str, str]]:
    """Load training examples from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def process_example(example: Dict[str, str], tokenizer) -> types.Datum:
    """
    Convert a training example into Tinker's Datum format.
    
    Args:
        example: Dictionary with 'task' and 'code' keys
        tokenizer: Tokenizer from the training client
    
    Returns:
        A Datum object ready for training
    """
    # Format the prompt with a clear task description
    prompt = f"Task: {example['task']}\n\nPython Code:\n"
    
    # Tokenize the prompt (input tokens, weight = 0)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)
    
    # Tokenize the code (completion tokens, weight = 1)
    # Add proper formatting and end token
    code_with_end = example['code'] + "\n\n"
    completion_tokens = tokenizer.encode(code_with_end, add_special_tokens=False)
    completion_weights = [1] * len(completion_tokens)
    
    # Combine tokens and weights
    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights
    
    # Shift for next-token prediction
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]
    
    # Create and return the Datum
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )

def visualize_example(datum: types.Datum, tokenizer, max_tokens: int = 30):
    """Visualize a training example for debugging."""
    input_tokens = datum.model_input.to_ints()
    target_tokens = datum.loss_fn_inputs['target_tokens'].tolist()
    weights = datum.loss_fn_inputs['weights'].tolist()
    
    print(f"\n{'Input Token':<30} {'Target Token':<30} {'Weight':<10}")
    print("-" * 70)
    
    for i, (inp, tgt, wgt) in enumerate(zip(input_tokens[:max_tokens], 
                                             target_tokens[:max_tokens], 
                                             weights[:max_tokens])):
        inp_str = repr(tokenizer.decode([inp]))[:28]
        tgt_str = repr(tokenizer.decode([tgt]))[:28]
        print(f"{inp_str:<30} {tgt_str:<30} {wgt:<10}")
    
    if len(input_tokens) > max_tokens:
        print(f"... ({len(input_tokens) - max_tokens} more tokens)")

def train_model(
    training_data_path: str = "data/training_examples.json",
    base_model: str = "Qwen/Qwen3-30B-A3B-Base",
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    model_name: str = "scientific-code-generator"
):
    """
    Main training function.
    
    Args:
        training_data_path: Path to training examples JSON file
        base_model: Base model to fine-tune
        num_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        model_name: Name to save the model under
    """
    # Check for API key
    if not os.getenv("TINKER_API_KEY"):
        raise ValueError("TINKER_API_KEY environment variable not set!")
    
    print("=" * 70)
    print("Scientific Code Generation Model Training")
    print("=" * 70)
    
    # Step 1: Create service client and check available models
    print("\n[1/5] Initializing Tinker service client...")
    service_client = tinker.ServiceClient()
    
    print("Available models:")
    for item in service_client.get_server_capabilities().supported_models:
        print(f"  - {item.model_name}")
    
    # Step 2: Create training client
    print(f"\n[2/5] Creating training client with base model: {base_model}")
    training_client = service_client.create_lora_training_client(
        base_model=base_model
    )
    tokenizer = training_client.get_tokenizer()
    print("✓ Training client created")
    
    # Step 3: Load and prepare training data
    print(f"\n[3/5] Loading training data from {training_data_path}...")
    examples = load_training_data(training_data_path)
    print(f"✓ Loaded {len(examples)} training examples")
    
    print("\nProcessing examples into Tinker format...")
    processed_examples = [process_example(ex, tokenizer) for ex in examples]
    print(f"✓ Processed {len(processed_examples)} examples")
    
    # Visualize first example
    print("\nVisualizing first training example:")
    visualize_example(processed_examples[0], tokenizer, max_tokens=25)
    
    # Step 4: Training loop
    print(f"\n[4/5] Starting training ({num_epochs} epochs)...")
    print("-" * 70)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Forward-backward pass
        fwdbwd_future = training_client.forward_backward(
            processed_examples, 
            "cross_entropy"
        )
        
        # Optimization step
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=learning_rate)
        )
        
        # Wait for results
        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()
        
        # Calculate and display loss
        logprobs = np.concatenate([
            output['logprobs'].tolist() 
            for output in fwdbwd_result.loss_fn_outputs
        ])
        weights = np.concatenate([
            example.loss_fn_inputs['weights'].tolist() 
            for example in processed_examples
        ])
        
        loss = -np.dot(logprobs, weights) / weights.sum()
        print(f"  Loss: {loss:.4f}")
    
    print("\n✓ Training completed!")
    
    # Step 5: Save model and create sampling client
    print(f"\n[5/5] Saving model weights as '{model_name}'...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=model_name
    )
    print("✓ Model saved and sampling client created")
    
    # Test the model with a sample query
    print("\n" + "=" * 70)
    print("Testing the trained model...")
    print("=" * 70)
    
    test_task = "Create a scatter plot of temperature vs pressure data with a linear regression line"
    test_prompt = f"Task: {test_task}\n\nPython Code:\n"
    
    prompt_tokens = tokenizer.encode(test_prompt, add_special_tokens=True)
    prompt = types.ModelInput.from_ints(prompt_tokens)
    
    sampling_params = types.SamplingParams(
        max_tokens=500,
        temperature=0.2,  # Lower temperature for more deterministic code
        stop=["\n\n\n"]  # Stop at triple newline
    )
    
    print(f"\nTest prompt: {test_task}")
    print("\nGenerating code...")
    
    future = sampling_client.sample(
        prompt=prompt,
        sampling_params=sampling_params,
        num_samples=3
    )
    result = future.result()
    
    print("\nGenerated code samples:")
    print("-" * 70)
    for i, seq in enumerate(result.sequences):
        generated_code = tokenizer.decode(seq.tokens)
        print(f"\nSample {i + 1}:")
        print(generated_code)
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("Training pipeline completed successfully!")
    print("=" * 70)
    print(f"\nModel saved as: {model_name}")
    print("You can now use this model for scientific code generation.")
    
    return sampling_client, tokenizer

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train a scientific code generation model with Tinker"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/training_examples.json",
        help="Path to training data JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Base",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="scientific-code-generator",
        help="Name for the saved model"
    )
    
    args = parser.parse_args()
    
    train_model(
        training_data_path=args.data,
        base_model=args.model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        model_name=args.name
    )

if __name__ == "__main__":
    main()

