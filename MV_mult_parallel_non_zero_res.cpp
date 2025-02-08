#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <cmath>
#include <Accelerate/Accelerate.h>
//// Commands to run the cpp file: clang++ -std=c++17 -o main MV_mult_threaded_nonzero_res.cpp -framework Accelerate->./main
// Constants for Theoretical Peak GFLOPS Calculation
const int NUM_CORES = 12;          // Number of cores for my mac with command:sysctl -n hw.ncpu
const double CLOCK_SPEED = 3.2;    // Clock speed in GHz:About this mac>System Report
const int FLOP_PER_CYCLE = 16;     // FLOPs per cycle per core

// Function to calculate theoretical peak GFLOPS
double CalculateTheoreticalPeakGFLOPS() {
    return NUM_CORES * CLOCK_SPEED * FLOP_PER_CYCLE;
}

// A small constant to account for floating-point precision errors
const double EPSILON = 1e-9;

// Function to create a square matrix with random values
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

// Function to create a random vector of given size
std::vector<double> CreateVector(int size) {
    std::vector<double> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (int i = 0; i < size; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}

// Parallel matrix-vector multiplication using tiling
void ParallelMatrixVectorMultiply(const std::vector<double>& matrix, 
                                  const std::vector<double>& vec, 
                                  std::vector<double>& result, 
                                  int startRow, int endRow, int size, int tileSize) {
    for (int i = startRow; i < endRow; i += tileSize) {
        for (int j = i; j < std::min(i + tileSize, endRow); j++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += matrix[j * size + k] * vec[k];
            }
            result[j] = sum;
        }
    }
}

// Function to handle parallel execution with tiling
std::vector<double> MatrixVectorMultiplyParallel(const std::vector<double>& matrix, 
                                                 const std::vector<double>& vec, 
                                                 int size, int numThreads, int tileSize) {
    std::vector<double> result(size, 0.0);
    std::vector<std::thread> threads;
    int chunkSize = size / numThreads;

    for (int t = 0; t < numThreads; t++) {
        int startRow = t * chunkSize;
        int endRow = (t == numThreads - 1) ? size : startRow + chunkSize;
        threads.emplace_back(ParallelMatrixVectorMultiply, std::ref(matrix), std::ref(vec), 
                             std::ref(result), startRow, endRow, size, tileSize);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return result;
}

// Perform matrix-vector multiplication using BLAS
std::vector<double> MatrixVectorMultiplyBLAS(const std::vector<double>& matrix, 
                                             const std::vector<double>& vec, 
                                             int size) {
    std::vector<double> result(size, 0.0);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1.0, matrix.data(), size, vec.data(), 1, 0.0, result.data(), 1);
    return result;
}

// Compute the residual matrix (difference between CPU result and BLAS result)
std::vector<double> ComputeResidualMatrix(const std::vector<double>& cpu_result, 
                                          const std::vector<double>& blas_result) {
    int size = cpu_result.size();
    std::vector<double> residual(size);
    for (int i = 0; i < size; ++i) {
        residual[i] = cpu_result[i] - blas_result[i];
    }
    return residual;
}

// Function to compute the residual norm (L2 norm of residual matrix)
double ComputeResidualNorm(const std::vector<double>& residual) {
    int size = residual.size();
    return cblas_dnrm2(size, residual.data(), 1);
}

int main() {
    // Output file to store execution data
    std::ofstream outFile("execution_times.txt");
    outFile << "Size,CPU Time,BLAS Time,CPU GFLOPS,BLAS GFLOPS,Theoretical Peak GFLOPS,Residual Norm\n";

    int dimensions[] = {10, 100, 200, 500, 1000, 2000, 4000, 5000, 8000, 10000};
    int numThreads = std::thread::hardware_concurrency(); // Detect available CPU threads
    int tileSize = 16; // Tile size for cache optimization
    double theoreticalPeakGFLOPS = CalculateTheoreticalPeakGFLOPS(); // Compute theoretical peak GFLOPS

    for (int size : dimensions) {
        // Generate random matrix and vector
        std::vector<double> matrix = CreateMatrix(size);
        std::vector<double> vec = CreateVector(size);

        // Measure parallel CPU execution time
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> cpu_result = MatrixVectorMultiplyParallel(matrix, vec, size, numThreads, tileSize);
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_time = stop - start;

        // Measure BLAS execution time
        start = std::chrono::high_resolution_clock::now();
        std::vector<double> blas_result = MatrixVectorMultiplyBLAS(matrix, vec, size);
        stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> blas_time = stop - start;

        // Compute GFLOPS
        double cpu_gflops = (2.0 * size * size) / (cpu_time.count() * 1e9);
        double blas_gflops = (2.0 * size * size) / (blas_time.count() * 1e9);

        // Compute residual matrix and its norm
        std::vector<double> residual = ComputeResidualMatrix(cpu_result, blas_result);
        double residual_norm = ComputeResidualNorm(residual);

        // Output results in the specified format
        std::cout << "----------------------------------------------------\n";
        std::cout << "Matrix-Vector Multiplication Test (Size: " << size << ")\n";
        std::cout << "CPU Time: " << cpu_time.count() << " sec, GFLOPS: " << cpu_gflops << "\n";
        std::cout << "BLAS Time: " << blas_time.count() << " sec, GFLOPS: " << blas_gflops << "\n";
        std::cout << "Theoretical Peak GFLOPS: " << theoreticalPeakGFLOPS << "\n";
        std::cout << "Residual Norm (||CPU result - BLAS result||_2): " << residual_norm << "\n";
        std::cout << "----------------------------------------------------\n";

        outFile << size << "," << cpu_time.count() << "," << blas_time.count() << ","
                << cpu_gflops << "," << blas_gflops << "," << theoreticalPeakGFLOPS << ","
                << residual_norm << "\n";
    }

    outFile.close();
    return 0;
}
