#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <cmath>
#include <Accelerate/Accelerate.h>
#include <iomanip>

// Constants for Theoretical Peak Performance Calculation
const int NUM_CORES = 12;              // Number of cores in your CPU
const double CLOCK_SPEED_GHZ = 3.2;   // Clock speed in GHz
const int FLOP_PER_CYCLE = 16;        // FLOPs per cycle per core

// Function to calculate theoretical peak performance (GFLOPS)
double CalculateTheoreticalPeakGFLOPS() {
    return NUM_CORES * CLOCK_SPEED_GHZ * FLOP_PER_CYCLE;
}

// Function to create a matrix with random values
std::vector<double> CreateMatrix(int size) {
    std::vector<double> matrix(size * size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (int i = 0; i < size * size; i++) {
        matrix[i] = dis(gen);
    }
    return matrix;
}

// Compute residual norm ||CPU result - BLAS result||_2
double ComputeResidualNorm(const std::vector<double>& cpu_result, const std::vector<double>& blas_result) {
    int size = cpu_result.size();
    std::vector<double> residual(size);
    for (int i = 0; i < size; ++i) {
        residual[i] = cpu_result[i] - blas_result[i];
    }
    return cblas_dnrm2(size, residual.data(), 1);
}

int main() {
    std::ofstream outFile("execution_results.txt");
    outFile << "Size,CPU Time,BLAS Time,CPU GFLOPS,BLAS GFLOPS,Theoretical Peak GFLOPS,Residual Norm\n";

    int dimensions[] = {100, 200, 500, 1000, 2000, 4000, 5000, 8000, 10000};
    double theoreticalPeakGFLOPS = CalculateTheoreticalPeakGFLOPS();

    for (int size : dimensions) {
        // Create random matrices A and B and result matrix C
        std::vector<double> A = CreateMatrix(size);
        std::vector<double> B = CreateMatrix(size);
        std::vector<double> C_cpu(size * size, 0.0);
        std::vector<double> C_blas(size * size, 0.0);

        // CPU matrix-matrix multiplication
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                double sum = 0.0;
                for (int k = 0; k < size; ++k) {
                    sum += A[i * size + k] * B[k * size + j];
                }
                C_cpu[i * size + j] = sum;
            }
        }
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_time = stop - start;

        // BLAS matrix-matrix multiplication
        start = std::chrono::high_resolution_clock::now();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, 1.0, A.data(), size, B.data(), size, 0.0, C_blas.data(), size);
        stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> blas_time = stop - start;

        // Compute GFLOPS
        double cpu_gflops = (2.0 * size * size * size) / (cpu_time.count() * 1e9);
        double blas_gflops = (2.0 * size * size * size) / (blas_time.count() * 1e9);

        // Compute residual norm
        double residual_norm = ComputeResidualNorm(C_cpu, C_blas);

        // Print results to console
        std::cout << "----------------------------------------------------\n";
        std::cout << "Matrix-Matrix Multiplication Test (Size: " << size << ")\n";
        std::cout << "CPU Time: " << cpu_time.count() << " sec, GFLOPS: " << cpu_gflops << "\n";
        std::cout << "BLAS Time: " << blas_time.count() << " sec, GFLOPS: " << blas_gflops << "\n";
        std::cout << "Theoretical Peak GFLOPS: " << theoreticalPeakGFLOPS << "\n";
        std::cout << "Residual Norm (||CPU result - BLAS result||_2): " << residual_norm << "\n";
        std::cout << "----------------------------------------------------\n";

        // Write results to file
        outFile << size << "," << cpu_time.count() << "," << blas_time.count() << ","
                << cpu_gflops << "," << blas_gflops << "," << theoreticalPeakGFLOPS << "," << residual_norm << "\n";
    }

    outFile.close();
    return 0;
}
