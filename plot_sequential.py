import pandas as pd
import matplotlib.pyplot as plt

# Load execution times from file
df = pd.read_csv("execution_times_seq.txt")

# Define color scheme
colors = {
    "CPU Time": "#1f77b4",   # Blue
    "BLAS Time": "#ff7f0e",  # Orange
    "CPU GFLOPS": "#2ca02c", # Green
    "BLAS GFLOPS": "#d62728", # Red
    "Theoretical GFLOPS": "#9467bd", # Purple
    "Residual Norm": "#8c564b" # Brown
}

# Function to format x-axis for better readability
def format_x_axis():
    plt.xticks(df["Size"], labels=[str(x) for x in df["Size"]], rotation=30, ha="right") 
    plt.xlabel("Matrix Size", fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()  # Adjust layout to prevent overlap

# Plot Execution Time: CPU vs BLAS
plt.figure(figsize=(9, 6))
plt.plot(df["Size"], df["CPU Time"], marker='o', color=colors["CPU Time"], linestyle='-', label="CPU Time (sec)")
plt.plot(df["Size"], df["BLAS Time"], marker='s', color=colors["BLAS Time"], linestyle='--', label="BLAS Time (sec)")
format_x_axis()
plt.ylabel("Execution Time (seconds)", fontsize=12)
plt.title("Execution Time: CPU vs BLAS", fontsize=14, fontweight='bold')
plt.legend()
plt.show()

# Plot FLOPS: CPU vs BLAS vs Theoretical Peak
plt.figure(figsize=(9, 6))
plt.plot(df["Size"], df["CPU GFLOPS"], marker='o', color=colors["CPU GFLOPS"], linestyle='-', label="CPU GFLOPS")
plt.plot(df["Size"], df["BLAS GFLOPS"], marker='s', color=colors["BLAS GFLOPS"], linestyle='--', label="BLAS GFLOPS")
plt.plot(df["Size"], df["Theoretical Peak GFLOPS"], marker='^', color=colors["Theoretical GFLOPS"], linestyle='-.', label="Theoretical Peak GFLOPS")
format_x_axis()
plt.ylabel("GFLOPS", fontsize=12)
plt.title("GFLOPS: CPU vs BLAS vs Theoretical Peak", fontsize=14, fontweight='bold')
plt.legend()
plt.show()

# Plot Residual Norm
plt.figure(figsize=(9, 6))
plt.plot(df["Size"], df["Residual Norm"], marker='o', color=colors["Residual Norm"], linestyle='-', label="Residual Norm ||CPU - BLAS||_2")
format_x_axis()
plt.ylabel("Residual Norm", fontsize=12)
plt.title("Residual Norm between CPU and BLAS Results", fontsize=14, fontweight='bold')
plt.legend()
plt.show()
