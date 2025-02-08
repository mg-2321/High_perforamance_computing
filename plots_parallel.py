import matplotlib.pyplot as plt
import pandas as pd

# Load data from the C++ output file
data = pd.read_csv("execution_times.txt")

# Print column names to verify correct reading
print("Columns in CSV:", data.columns)

# Ensure correct column references
if "Size" not in data.columns:
    raise ValueError("Column 'Size' is missing. Check execution_times.txt format.")

# Define valid matrix sizes for X-axis labels
valid_sizes = [10, 100, 200, 500, 1000, 2000, 4000, 5000, 8000, 10000]
data = data[data["Size"].isin(valid_sizes)]  # Filter valid sizes

# Define colors for clarity
colors = {
    "CPU Time": "#1f77b4",   # Blue
    "BLAS Time": "#ff7f0e",  # Orange
    "CPU GFLOPS": "#2ca02c",  # Green
    "BLAS GFLOPS": "#d62728",  # Red
    "Residual": "#9467bd",   # Purple
}

# Function to improve x-axis readability
def format_x_axis():
    plt.xticks(valid_sizes, labels=[str(x) for x in valid_sizes], rotation=30, ha="right")
    plt.xlabel("Matrix Size", fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()  # Adjust layout to prevent overlap

# -------------------- PLOTS --------------------

# 1️ **Execution Time Comparison (CPU vs BLAS)**
plt.figure(figsize=(9, 6))
plt.plot(data["Size"], data["CPU Time"], marker='o', color=colors["CPU Time"], linestyle='-', label="CPU Time (sec)")
plt.plot(data["Size"], data["BLAS Time"], marker='s', color=colors["BLAS Time"], linestyle='--', label="BLAS Time (sec)")
format_x_axis()
plt.ylabel("Execution Time (seconds)", fontsize=12)
plt.title("Execution Time: CPU vs BLAS", fontsize=14, fontweight='bold')
plt.legend()
plt.show()

# 2️ **Performance Comparison (GFLOPS: CPU vs BLAS)**
plt.figure(figsize=(9, 6))
plt.plot(data["Size"], data["CPU GFLOPS"], marker='o', color=colors["CPU GFLOPS"], linestyle='-', label="CPU GFLOPS")
plt.plot(data["Size"], data["BLAS GFLOPS"], marker='s', color=colors["BLAS GFLOPS"], linestyle='--', label="BLAS GFLOPS")
format_x_axis()
plt.ylabel("GFLOPS", fontsize=12)
plt.title("Performance Comparison: CPU vs BLAS (GFLOPS)", fontsize=14, fontweight='bold')
plt.legend()
plt.show()

# 3️ **Residuals (Numerical Error)**
plt.figure(figsize=(9, 6))
plt.plot(data["Size"], data["Residual Norm"], marker='o', color=colors["Residual"], linestyle='-', label="Residual Norm ||CPU - BLAS||_2")
format_x_axis()
plt.ylabel("Residual Norm", fontsize=12)
plt.title("Numerical Error (Residual Norm) Across Matrix Sizes", fontsize=14, fontweight='bold')
plt.legend()
plt.show()
