# Quick Start Guide

Get up and running with the Scientific Code Generation tutorial in 5 minutes!

## Prerequisites Check

Before starting, make sure you have:

- ✅ Python 3.8 or higher
- ✅ A Tinker API key ([Get one here](https://tinker-console.thinkingmachines.ai/))
- ✅ pip package manager

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Your API Key

**On macOS/Linux:**
```bash
export TINKER_API_KEY="your-api-key-here"
```

**On Windows:**
```cmd
set TINKER_API_KEY=your-api-key-here
```

**Or create a `.env` file:**
```bash
echo "TINKER_API_KEY=your-api-key-here" > .env
```

### 3. Generate Training Data

```bash
python generate_training_data.py
```

This creates `data/training_examples.json` with 20+ scientific computing examples.

### 4. Train the Model

```bash
python train_scientific_code_model.py
```

**Training options:**
```bash
# Custom number of epochs
python train_scientific_code_model.py --epochs 15

# Custom learning rate
python train_scientific_code_model.py --lr 5e-5

# Custom model name
python train_scientific_code_model.py --name my-scientific-model
```

### 5. Evaluate the Model

```bash
python evaluate_model.py
```

## Expected Output

### Training Output
```
======================================================================
Scientific Code Generation Model Training
======================================================================

[1/5] Initializing Tinker service client...
Available models:
  - meta-llama/Llama-3.1-70B
  - Qwen/Qwen3-30B-A3B-Base
  ...

[2/5] Creating training client...
✓ Training client created

[3/5] Loading training data...
✓ Loaded 20 training examples

[4/5] Starting training (10 epochs)...
Epoch 1/10
  Loss: 2.3456
Epoch 2/10
  Loss: 2.1234
...

[5/5] Saving model weights...
✓ Model saved and sampling client created
```

## Troubleshooting

### "TINKER_API_KEY not set"
Make sure you've exported the environment variable in your current terminal session.

### "Module not found: tinker"
Run `pip install tinker` or `pip install -r requirements.txt`

### "File not found: data/training_examples.json"
Run `python generate_training_data.py` first to create the training data.

### Training takes too long
- Reduce the number of epochs: `--epochs 5`
- Use a smaller base model (if available)
- Reduce the number of training examples

## Next Steps

1. **Experiment with different tasks**: Modify `generate_training_data.py` to add your own examples
2. **Try different models**: Change the `--model` parameter to use different base models
3. **Tune hyperparameters**: Experiment with learning rates and training epochs
4. **Extend the evaluation**: Add more test cases to `data/test_examples.json`

## Understanding the Code

- **`generate_training_data.py`**: Creates diverse scientific computing examples
- **`train_scientific_code_model.py`**: Main training script with Tinker API
- **`evaluate_model.py`**: Tests model performance on various tasks

## Cost Estimation

Training costs depend on:
- Number of training examples
- Number of epochs
- Model size
- Token count

Check [Tinker's pricing](https://tinker-console.thinkingmachines.ai/rate-card) for current rates.

## Getting Help

- Check the main [README.md](README.md) for detailed explanations
- Review [Tinker Documentation](https://docs.thinkingmachines.ai/)
- Join the [Tinker Discord](https://discord.gg/tinker) community

---

**Ready to start?** Run `python generate_training_data.py` now! 🚀

