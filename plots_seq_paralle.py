import pandas as pd
import matplotlib.pyplot as plt

# Load data
df_seq = pd.read_csv("execution_times_seq.txt")
df_par = pd.read_csv("execution_times.txt")

# Remove duplicates within each dataset
df_seq = df_seq.drop_duplicates(subset=["Size"]).reset_index(drop=True)
df_par = df_par.drop_duplicates(subset=["Size"]).reset_index(drop=True)

# Find common matrix sizes
common_sizes = sorted(set(df_seq["Size"]).intersection(set(df_par["Size"])))

# Filter both dataframes to only include common sizes
df_seq = df_seq[df_seq["Size"].isin(common_sizes)].reset_index(drop=True)
df_par = df_par[df_par["Size"].isin(common_sizes)].reset_index(drop=True)

# Ensure lengths are now the same
assert len(df_seq) == len(df_par), "Mismatch after filtering. Check dataset integrity."

# Define matrix sizes
sizes = df_seq["Size"].tolist()

# Debugging: Ensure we now have correct sizes
print("Final Common Sizes:", sizes)

# Define colors for better distinction
seq_color = "blue"
par_color = "red"

# Plot CPU Time comparison
plt.figure(figsize=(8, 6))
plt.plot(sizes, df_seq["CPU Time"], label="Sequential CPU Time", marker="o", linestyle="dashed", color=seq_color)
plt.plot(sizes, df_par["CPU Time"], label="Parallel CPU Time", marker="s", linestyle="solid", color=par_color)
plt.xlabel("Matrix Size")
plt.ylabel("Time (seconds)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.title("Execution Time: Sequential vs Parallel CPU")
plt.grid()
plt.show()

# Plot BLAS Time comparison
plt.figure(figsize=(8, 6))
plt.plot(sizes, df_seq["BLAS Time"], label="Sequential BLAS Time", marker="o", linestyle="dashed", color=seq_color)
plt.plot(sizes, df_par["BLAS Time"], label="Parallel BLAS Time", marker="s", linestyle="solid", color=par_color)
plt.xlabel("Matrix Size")
plt.ylabel("Time (seconds)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.title("BLAS Execution Time: Sequential vs Parallel")
plt.grid()
plt.show()

# Plot FLOPS comparison
plt.figure(figsize=(8, 6))
plt.plot(sizes, df_seq["CPU FLOPS"], label="Sequential CPU FLOPS", marker="o", linestyle="dashed", color=seq_color)
plt.plot(sizes, df_par["CPU FLOPS"], label="Parallel CPU FLOPS", marker="s", linestyle="solid", color=par_color)
plt.plot(sizes, df_seq["BLAS FLOPS"], label="Sequential BLAS FLOPS", marker="^", linestyle="dotted", color="green")
plt.plot(sizes, df_par["BLAS FLOPS"], label="Parallel BLAS FLOPS", marker="D", linestyle="dotted", color="purple")
plt.xlabel("Matrix Size")
plt.ylabel("FLOPS")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.title("FLOPS Comparison: Sequential vs Parallel")
plt.grid()
plt.show()

# Plot Residual comparison
plt.figure(figsize=(8, 6))
plt.plot(sizes, df_seq["Residual Norm"], label="Sequential Residual", marker="o", linestyle="dashed", color="red")
plt.plot(sizes, df_par["Residual Norm"], label="Parallel Residual", marker="s", linestyle="solid", color="blue")
plt.xlabel("Matrix Size")
plt.ylabel("Residual Norm ||CPU - BLAS||_2")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.title("Residual Norm: Sequential vs Parallel")
plt.grid()
plt.show()
