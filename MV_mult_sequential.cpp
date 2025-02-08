#include <iostream>
#include <fstream>  // For file writing
#include <random>
#include <vector>
#include <Accelerate/Accelerate.h>
#include <chrono>
#include <type_traits>
#include <thread>
// Commands to run the cpp file: clang++ -std=c++17 -o main MV_mult_sequential.cpp -framework Accelerate->./main
// Apple M4 Pro Specifications for Theoretical GFLOPS Calculation
const int NUM_CORES = 12;
const int FLOP_PER_CYCLE = 16;
const double CLOCK_SPEED_GHZ = 3.5; 

// Compute Theoretical Peak Performance
double ComputeTheoreticalPeakGFLOPS() {
    return NUM_CORES * CLOCK_SPEED_GHZ * FLOP_PER_CYCLE;
}

// Function to create a matrix with random values
template <typename T>
std::vector<T> CreateMatrix(int size) {
    std::vector<T> matrix(size * size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    for (int i = 0; i < size * size; i++) {
        matrix[i] = dis(gen);
    }
    return matrix;
}

// Function to create a random vector
template <typename T>
std::vector<T> CreateVector(int size) {
    std::vector<T> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    for (int i = 0; i < size; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}

// Function to perform matrix-vector multiplication using BLAS
template <typename T>
std::vector<T> MatrixVectorMultiplication(const std::vector<T>& matrix, const std::vector<T>& vec, int dim) {
    std::vector<T> result(dim, 0);
    
    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, dim, dim, 1.0f, matrix.data(), dim, vec.data(), 1, 0.0f, result.data(), 1);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgemv(CblasRowMajor, CblasNoTrans, dim, dim, 1.0, matrix.data(), dim, vec.data(), 1, 0.0, result.data(), 1);
    }
    
    return result;
}

// Compute residual matrix (CPU result - BLAS result)
template <typename T>
std::vector<T> ComputeResidualMatrix(const std::vector<T>& cpu_result, const std::vector<T>& blas_result) {
    int size = cpu_result.size();
    std::vector<T> residual(size);
    for (int i = 0; i < size; ++i) {
        residual[i] = cpu_result[i] - blas_result[i];
    }
    return residual;
}

// Compute residual norm ||CPU result - BLAS result||_2
template <typename T>
T ComputeResidualNorm(const std::vector<T>& residual) {
    int size = residual.size();
    if constexpr (std::is_same_v<T, float>) {
        return cblas_snrm2(size, residual.data(), 1);
    } else {
        return cblas_dnrm2(size, residual.data(), 1);
    }
}

// Function to run the test for both float and double precision
template <typename T>
void RunTest(int dim, std::ofstream& outFile, std::ofstream& residualFile) {
    std::cout << "----------------------------------------------------\n";
    std::cout << "Matrix-Vector Multiplication Test (Size: " << dim << ", Precision: " 
              << (std::is_same_v<T, float> ? "float" : "double") << ")\n";

    // Create matrix and vector
    std::vector<T> matrix = CreateMatrix<T>(dim);
    std::vector<T> vec = CreateVector<T>(dim);

    // Compute CPU sequential multiplication
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<T> cpu_result = MatrixVectorMultiplication(matrix, vec, dim);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = stop - start;

    // Compute BLAS multiplication
    start = std::chrono::high_resolution_clock::now();
    std::vector<T> blas_result = MatrixVectorMultiplication(matrix, vec, dim);
    stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> blas_time = stop - start;

    // Compute GFLOPS
    double cpu_gflops = (2.0 * dim * dim) / (cpu_time.count() * 1e9);
    double blas_gflops = (2.0 * dim * dim) / (blas_time.count() * 1e9);

    // Compute theoretical peak performance
    double theoretical_peak_gflops = ComputeTheoreticalPeakGFLOPS();

    // Compute residual
    std::vector<T> residual = ComputeResidualMatrix(cpu_result, blas_result);
    T residual_norm = ComputeResidualNorm(residual);

    // Output results to console
    std::cout << "CPU Time: " << cpu_time.count() << " sec, GFLOPS: " << cpu_gflops << "\n";
    std::cout << "BLAS Time: " << blas_time.count() << " sec, GFLOPS: " << blas_gflops << "\n";
    std::cout << "Theoretical Peak GFLOPS: " << theoretical_peak_gflops << "\n";
    std::cout << "Residual (||CPU result - BLAS result||_2): " << residual_norm << "\n";
    std::cout << "----------------------------------------------------\n";

    // Write results to file for plotting
    outFile << dim << "," << cpu_time.count() << "," << blas_time.count() << "," 
            << cpu_gflops << "," << blas_gflops << "," << theoretical_peak_gflops << "," << residual_norm << "\n";

    // Write full residual matrix to file
    residualFile << "Residual Matrix (Size: " << dim << ", Precision: " 
                 << (std::is_same_v<T, float> ? "float" : "double") << ")\n";
    for (int i = 0; i < dim; ++i) {
        residualFile << residual[i] << "\n";
    }
    residualFile << "----------------------------------------------------\n";
}

int main() {
    std::ofstream outFile("execution_times_seq.txt");
    outFile << "Size,CPU Time,BLAS Time,CPU GFLOPS,BLAS GFLOPS,Theoretical Peak GFLOPS,Residual Norm\n";

    std::ofstream residualFile("residuals.txt");

    // Explicitly test n = 100
    RunTest<float>(100, outFile, residualFile);
    RunTest<double>(100, outFile, residualFile);

    // Additional tests for different sizes
    int dimensions[] = {10, 100, 200, 500, 1000, 2000, 4000, 5000, 8000, 10000};

    for (int dim : dimensions) {
        RunTest<float>(dim, outFile, residualFile);
        RunTest<double>(dim, outFile, residualFile);
    }

    outFile.close();
    residualFile.close();
    return 0;
}
