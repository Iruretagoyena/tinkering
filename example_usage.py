"""
Example script showing how to use a trained scientific code generation model.

This demonstrates how to load a saved model and generate code for various tasks.
"""

import os
import tinker
from tinker import types

def generate_scientific_code(task_description: str, model_name: str = "scientific-code-generator"):
    """
    Generate Python code for a scientific computing task.
    
    Args:
        task_description: Natural language description of the task
        model_name: Name of the saved model
    
    Returns:
        Generated Python code as a string
    """
    # Check for API key
    if not os.getenv("TINKER_API_KEY"):
        raise ValueError("TINKER_API_KEY environment variable not set!")
    
    # Initialize service client
    service_client = tinker.ServiceClient()
    
    # Load the model (this is a simplified example)
    # In practice, you would load your saved model here
    # For demonstration, we'll show the structure
    
    print(f"Generating code for: {task_description}")
    print("-" * 70)
    
    # This is a placeholder - actual implementation depends on how
    # Tinker handles loading saved models
    print("\nNote: To use a saved model, you need to:")
    print("1. Load the model using Tinker's model loading API")
    print("2. Create a sampling client from the loaded model")
    print("3. Use the sampling client to generate code")
    
    # Example structure (actual API may differ):
    # sampling_client = service_client.load_saved_model(model_name)
    # tokenizer = sampling_client.get_tokenizer()
    # 
    # prompt = f"Task: {task_description}\n\nPython Code:\n"
    # prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    # prompt_input = types.ModelInput.from_ints(prompt_tokens)
    # 
    # sampling_params = types.SamplingParams(
    #     max_tokens=500,
    #     temperature=0.2,
    #     stop=["\n\n\n"]
    # )
    # 
    # future = sampling_client.sample(
    #     prompt=prompt_input,
    #     sampling_params=sampling_params,
    #     num_samples=1
    # )
    # result = future.result()
    # 
    # if result.sequences:
    #     generated_code = tokenizer.decode(result.sequences[0].tokens)
    #     return generated_code
    
    return None

def main():
    """Example usage of the trained model."""
    
    example_tasks = [
        "Create a scatter plot of temperature vs pressure data with a linear regression line",
        "Calculate the mean and standard deviation of a dataset",
        "Perform a t-test comparing two groups of measurements",
        "Integrate the function f(x) = x² * sin(x) from 0 to π",
        "Load a CSV file, remove missing values, and calculate column means"
    ]
    
    print("=" * 70)
    print("Scientific Code Generation - Example Usage")
    print("=" * 70)
    
    for i, task in enumerate(example_tasks, 1):
        print(f"\nExample {i}:")
        code = generate_scientific_code(task)
        
        if code:
            print("\nGenerated Code:")
            print(code)
        else:
            print("\n(Code generation would happen here)")
        
        print("\n" + "=" * 70)
    
    print("\nTo use this script with your trained model:")
    print("1. Make sure your model is saved and accessible")
    print("2. Update the model loading code based on Tinker's API")
    print("3. Run: python example_usage.py")

if __name__ == "__main__":
    main()

