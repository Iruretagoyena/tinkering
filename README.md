# Tinkering with Tinker: Scientific Code Generation Tutorial

A comprehensive tutorial on fine-tuning LLMs with Tinker to build a **Scientific Code Generation Assistant** - a model that generates Python code for common scientific computing tasks.

## 🎯 Project Overview

This tutorial teaches you how to use [Tinker](https://thinkingmachines.ai/) from Thinking Machines to fine-tune a language model that can generate Python code for scientific tasks like:
- Data visualization and analysis
- Statistical computations
- Scientific simulations
- Data processing and cleaning
- Numerical methods

**Why this use case?**
- ✅ **Real-world application**: Scientists and researchers need code generation tools
- ✅ **ML Engineering showcase**: Demonstrates proper data preparation, training, and evaluation
- ✅ **Educational**: Teaches both Tinker API and scientific computing best practices
- ✅ **Practical**: The fine-tuned model can be used immediately in research workflows

## 📚 What You'll Learn

1. **Tinker API Fundamentals**
   - Setting up the Tinker client
   - Creating training clients with LoRA
   - Data preparation and formatting
   - Training loops and optimization
   - Model sampling and evaluation

2. **ML Engineering Best Practices**
   - Proper data preprocessing
   - Loss function configuration
   - Training monitoring
   - Model evaluation strategies

3. **Scientific Computing**
   - Common Python libraries (NumPy, SciPy, Matplotlib, Pandas)
   - Scientific code patterns
   - Best practices for reproducible research

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Tinker API key ([get one here](https://tinker-console.thinkingmachines.ai/))
- Basic understanding of Python and machine learning

### Installation

```bash
# Clone or navigate to this directory
cd Tinkering

# Install dependencies
pip install -r requirements.txt

# Set your Tinker API key
export TINKER_API_KEY="your-api-key-here"
```

### Run the Tutorial

```bash
# Step 1: Generate training data
python generate_training_data.py

# Step 2: Train the model
python train_scientific_code_model.py

# Step 3: Evaluate the model
python evaluate_model.py
```

## 📖 Tutorial Structure

### Part 1: Understanding the Problem

We want to fine-tune a model that takes natural language descriptions of scientific tasks and generates corresponding Python code. For example:

**Input:**
```
Task: Create a scatter plot of temperature vs pressure data with a linear regression line
```

**Output:**
```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Assuming data is in arrays: temperature, pressure
slope, intercept, r_value, p_value, std_err = stats.linregress(temperature, pressure)
line = slope * temperature + intercept

plt.figure(figsize=(10, 6))
plt.scatter(temperature, pressure, alpha=0.6, label='Data points')
plt.plot(temperature, line, 'r-', label=f'Linear fit (R²={r_value**2:.3f})')
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (kPa)')
plt.title('Temperature vs Pressure with Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Part 2: Data Preparation

We'll create a diverse dataset covering:
- **Data Visualization**: Plots, histograms, heatmaps
- **Statistical Analysis**: Hypothesis testing, correlation analysis
- **Numerical Methods**: Integration, optimization, curve fitting
- **Data Processing**: Cleaning, transformation, aggregation

See `generate_training_data.py` for the complete data generation logic.

### Part 3: Training with Tinker

The training process involves:
1. **Creating a Training Client**: Initialize with a base model
2. **Preparing Data**: Convert examples to Tinker's `Datum` format
3. **Training Loop**: Forward-backward passes and optimization steps
4. **Monitoring**: Track loss and model performance

### Part 4: Evaluation

We evaluate the model on:
- **Code correctness**: Does the generated code run without errors?
- **Task completion**: Does it solve the requested problem?
- **Code quality**: Is it well-structured and follows best practices?

## 📁 Project Structure

```
Tinkering/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── generate_training_data.py      # Script to create training examples
├── train_scientific_code_model.py # Main training script
├── evaluate_model.py              # Model evaluation script
└── data/
    ├── training_examples.json     # Generated training data
    └── test_examples.json         # Test cases for evaluation
```

## 🔬 Example Use Cases

### 1. Data Visualization
**Task**: "Plot a histogram of exam scores with 20 bins and add a normal distribution overlay"

### 2. Statistical Analysis
**Task**: "Perform a t-test comparing two groups of measurements and report the p-value"

### 3. Numerical Methods
**Task**: "Integrate the function f(x) = x² * sin(x) from 0 to π using Simpson's rule"

### 4. Data Processing
**Task**: "Load a CSV file, remove rows with missing values, and calculate the mean of each column"

## 🎓 Key Concepts Explained

### LoRA (Low-Rank Adaptation)
Tinker uses LoRA for efficient fine-tuning, allowing us to adapt large models with minimal computational cost.

### Loss Functions
We use cross-entropy loss with proper weighting to focus learning on the code generation task.

### Tokenization and Formatting
Proper tokenization ensures the model learns the structure of Python code effectively.

## 📊 Expected Results

After training, you should see:
- **Training Loss**: Decreasing over epochs
- **Code Generation**: Model produces syntactically correct Python code
- **Task Accuracy**: Model solves 70-80% of test cases correctly

## 🛠️ Advanced Topics

- **Hyperparameter Tuning**: Learning rates, batch sizes, LoRA ranks
- **Data Augmentation**: Expanding the training dataset
- **Multi-task Learning**: Training on multiple scientific domains
- **Deployment**: Integrating the model into a code editor or IDE

## 📚 Additional Resources

- [Tinker Documentation](https://docs.thinkingmachines.ai/)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [Scientific Python Ecosystem](https://scipy.org/)

## 🤝 Contributing

Feel free to:
- Add more training examples
- Improve evaluation metrics
- Extend to other scientific domains
- Share your results!

## 📝 License

This tutorial is provided as-is for educational purposes.

## 🙏 Acknowledgments

- Thinking Machines for building Tinker
- The scientific Python community for excellent tools
- All contributors to open-source ML and scientific computing

---

**Ready to start?** Begin with `generate_training_data.py` to create your dataset!
