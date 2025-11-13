"""
Generate training data for scientific code generation model.

This script creates diverse examples of scientific computing tasks
paired with their Python code solutions.
"""

import json
from typing import List, Dict

def generate_training_examples() -> List[Dict[str, str]]:
    """
    Generate training examples for scientific code generation.
    Each example contains a task description and corresponding Python code.
    """
    examples = []
    
    # Data Visualization Examples
    examples.extend([
        {
            "task": "Create a scatter plot of temperature vs pressure data with a linear regression line",
            "code": """import matplotlib.pyplot as plt
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
plt.show()"""
        },
        {
            "task": "Plot a histogram of exam scores with 20 bins and add a normal distribution overlay",
            "code": """import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Assuming scores is an array of exam scores
mu, sigma = np.mean(scores), np.std(scores)
x = np.linspace(scores.min(), scores.max(), 100)
normal_curve = stats.norm.pdf(x, mu, sigma) * len(scores) * (scores.max() - scores.min()) / 20

plt.figure(figsize=(10, 6))
plt.hist(scores, bins=20, alpha=0.7, label='Score distribution', edgecolor='black')
plt.plot(x, normal_curve, 'r-', linewidth=2, label=f'Normal fit (μ={mu:.1f}, σ={sigma:.1f})')
plt.xlabel('Exam Score')
plt.ylabel('Frequency')
plt.title('Distribution of Exam Scores')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""
        },
        {
            "task": "Create a heatmap of correlation matrix for a pandas DataFrame",
            "code": """import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming df is a pandas DataFrame
correlation_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()"""
        },
        {
            "task": "Plot multiple time series on the same graph with different colors and a legend",
            "code": """import matplotlib.pyplot as plt
import pandas as pd

# Assuming df is a DataFrame with datetime index and multiple columns
plt.figure(figsize=(12, 6))
for column in df.columns:
    plt.plot(df.index, df[column], label=column, linewidth=2)

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""
        }
    ])
    
    # Statistical Analysis Examples
    examples.extend([
        {
            "task": "Perform a t-test comparing two groups of measurements and report the p-value",
            "code": """from scipy import stats
import numpy as np

# Assuming group1 and group2 are arrays of measurements
t_statistic, p_value = stats.ttest_ind(group1, group2)

print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant at α=0.05: {p_value < 0.05}")"""
        },
        {
            "task": "Calculate Pearson correlation coefficient and its p-value between two variables",
            "code": """from scipy import stats
import numpy as np

# Assuming x and y are arrays of paired measurements
correlation, p_value = stats.pearsonr(x, y)

print(f"Pearson correlation coefficient: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant correlation: {p_value < 0.05}")"""
        },
        {
            "task": "Perform ANOVA test on three groups of data",
            "code": """from scipy import stats
import numpy as np

# Assuming group1, group2, group3 are arrays of measurements
f_statistic, p_value = stats.f_oneway(group1, group2, group3)

print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant difference between groups: {p_value < 0.05}")"""
        },
        {
            "task": "Calculate confidence interval for the mean of a dataset",
            "code": """from scipy import stats
import numpy as np

# Assuming data is an array of measurements
confidence_level = 0.95
n = len(data)
mean = np.mean(data)
std_err = stats.sem(data)
h = std_err * stats.t.ppf((1 + confidence_level) / 2, n - 1)

print(f"Mean: {mean:.4f}")
print(f"{confidence_level*100}% Confidence Interval: [{mean - h:.4f}, {mean + h:.4f}]")"""
        }
    ])
    
    # Numerical Methods Examples
    examples.extend([
        {
            "task": "Integrate the function f(x) = x² * sin(x) from 0 to π using Simpson's rule",
            "code": """from scipy import integrate
import numpy as np

def f(x):
    return x**2 * np.sin(x)

result, error = integrate.quad(f, 0, np.pi)
print(f"Integral value: {result:.6f}")
print(f"Estimated error: {error:.2e}")"""
        },
        {
            "task": "Find the minimum of the function f(x) = x⁴ - 5x² + 4 using optimization",
            "code": """from scipy.optimize import minimize_scalar
import numpy as np

def f(x):
    return x**4 - 5*x**2 + 4

result = minimize_scalar(f, method='brent')
print(f"Minimum value: {result.fun:.6f}")
print(f"At x = {result.x:.6f}")"""
        },
        {
            "task": "Solve a system of linear equations Ax = b",
            "code": """import numpy as np
from scipy.linalg import solve

# Assuming A is a matrix and b is a vector
# A = np.array([[1, 2], [3, 4]])
# b = np.array([5, 6])
x = solve(A, b)
print(f"Solution: {x}")"""
        },
        {
            "task": "Fit a polynomial of degree 3 to data points and plot the result",
            "code": """import numpy as np
import matplotlib.pyplot as plt

# Assuming x_data and y_data are arrays of data points
coefficients = np.polyfit(x_data, y_data, 3)
polynomial = np.poly1d(coefficients)
x_fit = np.linspace(x_data.min(), x_data.max(), 100)
y_fit = polynomial(x_fit)

plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Data points', alpha=0.6)
plt.plot(x_fit, y_fit, 'r-', label='Polynomial fit (degree 3)', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Curve Fitting')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""
        }
    ])
    
    # Data Processing Examples
    examples.extend([
        {
            "task": "Load a CSV file, remove rows with missing values, and calculate the mean of each column",
            "code": """import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('data.csv')

# Remove rows with missing values
df_clean = df.dropna()

# Calculate mean of each column
column_means = df_clean.mean()
print("Mean of each column:")
print(column_means)"""
        },
        {
            "task": "Normalize a dataset to have zero mean and unit variance",
            "code": """import numpy as np
from sklearn.preprocessing import StandardScaler

# Assuming data is a 2D array (samples x features)
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

print(f"Original mean: {np.mean(data, axis=0)}")
print(f"Normalized mean: {np.mean(data_normalized, axis=0)}")
print(f"Normalized std: {np.std(data_normalized, axis=0)}")"""
        },
        {
            "task": "Group data by category and calculate summary statistics for each group",
            "code": """import pandas as pd

# Assuming df is a DataFrame with a 'category' column and a 'value' column
summary = df.groupby('category')['value'].agg(['mean', 'std', 'min', 'max', 'count'])
print(summary)"""
        },
        {
            "task": "Filter a DataFrame to keep only rows where a column value is above a threshold",
            "code": """import pandas as pd

# Assuming df is a DataFrame and 'column_name' is the column to filter on
threshold = 100
filtered_df = df[df['column_name'] > threshold]
print(f"Original rows: {len(df)}")
print(f"Filtered rows: {len(filtered_df)}")"""
        }
    ])
    
    # Scientific Computing Examples
    examples.extend([
        {
            "task": "Generate a sine wave signal, add noise, and apply a low-pass filter",
            "code": """import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Generate sine wave
t = np.linspace(0, 1, 1000, False)
frequency = 5  # Hz
sine_wave = np.sin(2 * np.pi * frequency * t)

# Add noise
noise = np.random.normal(0, 0.1, len(sine_wave))
noisy_signal = sine_wave + noise

# Apply low-pass filter
b, a = signal.butter(4, 0.1, 'low')
filtered_signal = signal.filtfilt(b, a, noisy_signal)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(t, sine_wave, label='Original', alpha=0.7)
plt.plot(t, noisy_signal, label='Noisy', alpha=0.5)
plt.plot(t, filtered_signal, label='Filtered', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Signal Filtering')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""
        },
        {
            "task": "Calculate the Fast Fourier Transform of a signal and plot the power spectrum",
            "code": """import numpy as np
import matplotlib.pyplot as plt

# Assuming signal is an array of signal values and sample_rate is the sampling frequency
fft_values = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), 1/sample_rate)
power_spectrum = np.abs(fft_values)**2

# Plot only positive frequencies
positive_freq_idx = frequencies >= 0
plt.figure(figsize=(10, 6))
plt.plot(frequencies[positive_freq_idx], power_spectrum[positive_freq_idx])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Power Spectrum')
plt.grid(True, alpha=0.3)
plt.show()"""
        },
        {
            "task": "Solve a differential equation using numerical integration",
            "code": """from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def dydt(y, t):
    # Example: dy/dt = -y (exponential decay)
    return -y

# Initial condition and time points
y0 = 1.0
t = np.linspace(0, 10, 100)

# Solve the ODE
solution = odeint(dydt, y0, t)

plt.figure(figsize=(10, 6))
plt.plot(t, solution, linewidth=2)
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('Solution of Differential Equation')
plt.grid(True, alpha=0.3)
plt.show()"""
        }
    ])
    
    return examples

def save_training_data(examples: List[Dict[str, str]], filename: str = "data/training_examples.json"):
    """Save training examples to a JSON file."""
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"Saved {len(examples)} training examples to {filename}")

def main():
    """Generate and save training data."""
    print("Generating training examples for scientific code generation...")
    examples = generate_training_examples()
    save_training_data(examples)
    print(f"\nGenerated {len(examples)} examples covering:")
    print("  - Data Visualization")
    print("  - Statistical Analysis")
    print("  - Numerical Methods")
    print("  - Data Processing")
    print("  - Scientific Computing")

if __name__ == "__main__":
    main()

