"""
Evaluate the trained scientific code generation model.

This script tests the model on various scientific computing tasks
and evaluates code correctness, task completion, and code quality.
"""

import json
import os
import subprocess
import tempfile
from typing import List, Dict, Tuple
import tinker
from tinker import types

def load_test_data(filename: str = "data/test_examples.json") -> List[Dict[str, str]]:
    """Load test examples from JSON file."""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Creating sample test data...")
        return create_sample_test_data()
    
    with open(filename, 'r') as f:
        return json.load(f)

def create_sample_test_data() -> List[Dict[str, str]]:
    """Create sample test data if test file doesn't exist."""
    return [
        {
            "task": "Create a simple line plot of x from 0 to 10 and y = x^2",
            "expected_keywords": ["matplotlib", "plot", "x**2", "x^2"]
        },
        {
            "task": "Calculate the mean and standard deviation of a list of numbers",
            "expected_keywords": ["mean", "std", "numpy", "np."]
        },
        {
            "task": "Load a CSV file using pandas",
            "expected_keywords": ["pandas", "read_csv", "pd."]
        }
    ]

def load_model(service_client, model_name: str = "scientific-code-generator"):
    """Load a saved model for sampling."""
    # Note: In Tinker, you typically access saved models through the service client
    # This is a placeholder - actual implementation depends on Tinker's API
    print(f"Loading model: {model_name}")
    # You may need to use a different method based on Tinker's API
    # For now, we'll assume the model can be accessed via sampling client
    return None

def generate_code(
    sampling_client,
    tokenizer,
    task: str,
    max_tokens: int = 500,
    temperature: float = 0.2
) -> str:
    """Generate code for a given task."""
    prompt = f"Task: {task}\n\nPython Code:\n"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_input = types.ModelInput.from_ints(prompt_tokens)
    
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["\n\n\n"]
    )
    
    future = sampling_client.sample(
        prompt=prompt_input,
        sampling_params=sampling_params,
        num_samples=1
    )
    result = future.result()
    
    if result.sequences:
        return tokenizer.decode(result.sequences[0].tokens)
    return ""

def check_syntax(code: str) -> Tuple[bool, str]:
    """
    Check if generated code has valid Python syntax.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        compile(code, '<string>', 'exec')
        return True, ""
    except SyntaxError as e:
        return False, str(e)

def check_imports(code: str) -> List[str]:
    """Extract import statements from code."""
    imports = []
    for line in code.split('\n'):
        line = line.strip()
        if line.startswith('import ') or line.startswith('from '):
            imports.append(line)
    return imports

def check_keywords(code: str, keywords: List[str]) -> int:
    """Check how many expected keywords appear in the code."""
    code_lower = code.lower()
    found = sum(1 for keyword in keywords if keyword.lower() in code_lower)
    return found

def run_code_safely(code: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Attempt to run code in a safe environment.
    
    Returns:
        (success, output_or_error)
    """
    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        # Try to run the code with a timeout
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Code execution timed out"
    except Exception as e:
        return False, str(e)
    finally:
        # Clean up
        try:
            os.unlink(temp_file)
        except:
            pass

def evaluate_example(
    sampling_client,
    tokenizer,
    example: Dict[str, str],
    verbose: bool = True
) -> Dict:
    """
    Evaluate a single test example.
    
    Returns:
        Dictionary with evaluation metrics
    """
    task = example['task']
    expected_keywords = example.get('expected_keywords', [])
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Task: {task}")
        print(f"{'='*70}")
    
    # Generate code
    generated_code = generate_code(sampling_client, tokenizer, task)
    
    if verbose:
        print("\nGenerated Code:")
        print("-" * 70)
        print(generated_code)
        print("-" * 70)
    
    # Evaluate
    syntax_valid, syntax_error = check_syntax(generated_code)
    imports = check_imports(generated_code)
    keyword_matches = check_keywords(generated_code, expected_keywords) if expected_keywords else 0
    keyword_score = keyword_matches / len(expected_keywords) if expected_keywords else 0
    
    # Try to run the code (optional, may fail due to missing data/files)
    code_runs = False
    run_error = ""
    if syntax_valid:
        code_runs, run_error = run_code_safely(generated_code)
    
    evaluation = {
        'task': task,
        'generated_code': generated_code,
        'syntax_valid': syntax_valid,
        'syntax_error': syntax_error,
        'imports': imports,
        'keyword_matches': keyword_matches,
        'keyword_score': keyword_score,
        'code_runs': code_runs,
        'run_error': run_error,
        'code_length': len(generated_code)
    }
    
    if verbose:
        print(f"\nEvaluation Results:")
        print(f"  Syntax Valid: {syntax_valid}")
        if not syntax_valid:
            print(f"  Syntax Error: {syntax_error}")
        print(f"  Imports Found: {len(imports)}")
        if imports:
            for imp in imports:
                print(f"    - {imp}")
        if expected_keywords:
            print(f"  Keyword Matches: {keyword_matches}/{len(expected_keywords)} ({keyword_score*100:.1f}%)")
        print(f"  Code Runs: {code_runs}")
        if not code_runs and run_error:
            print(f"  Run Error: {run_error[:200]}")
    
    return evaluation

def evaluate_model(
    model_name: str = "scientific-code-generator",
    test_data_path: str = "data/test_examples.json",
    verbose: bool = True
):
    """Main evaluation function."""
    print("=" * 70)
    print("Scientific Code Generation Model Evaluation")
    print("=" * 70)
    
    # Check for API key
    if not os.getenv("TINKER_API_KEY"):
        raise ValueError("TINKER_API_KEY environment variable not set!")
    
    # Initialize service client
    print("\n[1/3] Initializing Tinker service client...")
    service_client = tinker.ServiceClient()
    
    # Load test data
    print(f"\n[2/3] Loading test data from {test_data_path}...")
    test_examples = load_test_data(test_data_path)
    print(f"✓ Loaded {len(test_examples)} test examples")
    
    # For evaluation, we need to load the saved model
    # This is a simplified version - actual implementation may vary
    print(f"\n[3/3] Loading model: {model_name}...")
    print("Note: Model loading depends on Tinker's API for saved models")
    print("For now, you'll need to recreate the sampling client from training")
    
    # In a real scenario, you would load the saved model here
    # For this tutorial, we'll show the evaluation structure
    print("\n" + "=" * 70)
    print("Evaluation Structure")
    print("=" * 70)
    print("\nTo evaluate your trained model:")
    print("1. Load the model using the same method as in training")
    print("2. Run evaluate_example() for each test case")
    print("3. Aggregate results across all test cases")
    
    # Show example evaluation structure
    print("\nExample evaluation metrics to track:")
    print("  - Syntax validity rate")
    print("  - Keyword match rate")
    print("  - Code execution success rate")
    print("  - Average code length")
    print("  - Import statement quality")
    
    # If we had a model, we would do:
    # evaluations = []
    # for example in test_examples:
    #     eval_result = evaluate_example(sampling_client, tokenizer, example, verbose)
    #     evaluations.append(eval_result)
    # 
    # # Aggregate results
    # syntax_rate = sum(e['syntax_valid'] for e in evaluations) / len(evaluations)
    # keyword_rate = sum(e['keyword_score'] for e in evaluations) / len(evaluations)
    # ...
    
    print("\n" + "=" * 70)
    print("Evaluation framework ready!")
    print("=" * 70)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate a trained scientific code generation model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="scientific-code-generator",
        help="Name of the saved model"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/test_examples.json",
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed evaluation output"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_name=args.model,
        test_data_path=args.test_data,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()

