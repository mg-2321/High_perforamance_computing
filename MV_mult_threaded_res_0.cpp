#include <iostream>
#include <fstream>  // For writing results to a file
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <cmath>
#include <iomanip>  // For formatting output
#include <Accelerate/Accelerate.h>
// Commands to run the cpp file: clang++ -std=c++17 -o main MV_mult_threaded_res_0.cpp -framework Accelerate->./main
// Constants for Theoretical Peak Performance Calculation
const int NUM_CORES = 12;              // Update with your CPU's number of cores
const double CLOCK_SPEED_GHZ = 3.5;   // Update with your CPU's clock speed in GHz
const int FLOP_PER_CYCLE = 16;        // Update based on double precision FLOP/cycle (e.g., FMA)

// Function to calculate theoretical peak performance (GFLOPS)
double CalculateTheoreticalPeakGFLOPS() {
    return NUM_CORES * CLOCK_SPEED_GHZ * FLOP_PER_CYCLE;
}

int main() {
    // Open TXT file for writing results
    std::ofstream txtFile("execution_parallel.txt");

    // Print headers to the file
    txtFile << "Matrix-Vector Multiplication Performance\n";
    txtFile << "----------------------------------------------------------------------------------\n";
    txtFile << std::setw(10) << "Size"
            << std::setw(15) << "CPU Time (s)"
            << std::setw(15) << "BLAS Time (s)"
            << std::setw(15) << "CPU GFLOPS"
            << std::setw(15) << "BLAS GFLOPS"
            << std::setw(15) << "Theo. GFLOPS"
            << std::setw(15) << "Residual Norm\n";
    txtFile << "----------------------------------------------------------------------------------\n";

    // Define problem sizes
    int dimensions[] = {10, 100, 200, 500, 1000, 2000, 4000, 5000, 8000, 10000};
    int numThreads = std::thread::hardware_concurrency();
    int tileSize = 16;

    // Calculate the theoretical peak GFLOPS
    double theoreticalPeakGFLOPS = CalculateTheoreticalPeakGFLOPS();

    for (int size : dimensions) {
        // Create matrix and vector
        std::vector<double> matrix(size * size);
        std::vector<double> vec(size);

        // Measure CPU execution time
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> cpu_result(size, 0.0);  // Replace with actual computation
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_time = stop - start;

        // Measure BLAS execution time
        start = std::chrono::high_resolution_clock::now();
        std::vector<double> blas_result(size, 0.0); // Replace with actual computation
        stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> blas_time = stop - start;

        // Compute GFLOPS
        double cpu_gflops = (2.0 * size * size) / (cpu_time.count() * 1e9);
        double blas_gflops = (2.0 * size * size) / (blas_time.count() * 1e9);

        // Compute residual norm (dummy value here)
        double residual_norm = 0.0;

        // Print results to the text file
        txtFile << std::setw(10) << size
                << std::setw(15) << cpu_time.count()
                << std::setw(15) << blas_time.count()
                << std::setw(15) << cpu_gflops
                << std::setw(15) << blas_gflops
                << std::setw(15) << theoreticalPeakGFLOPS
                << std::setw(15) << residual_norm << "\n";
    }

    txtFile << "----------------------------------------------------------------------------------\n";
    txtFile.close();  // Close file

    std::cout << "Results saved to execution_parallel.txt\n";
    return 0;
}
