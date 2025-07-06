#include "hip_utils.h"
#include "timer.h"
#include "helper_functions.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <cmath>

// Function declarations from HIP kernel file
extern "C" {
    void launchMonteCarloPi(unsigned int *results, int samples_per_thread, unsigned int seed,
                           int blockSize, int gridSize);
    void launchMonteCarloIntegration(float *results, int samples_per_thread, unsigned int seed,
                                    int blockSize, int gridSize);
    void launchMonteCarloOptionPricing(float *results, float S0, float K, float T, float r, float sigma,
                                      int samples_per_thread, unsigned int seed,
                                      int blockSize, int gridSize);
    void launchMonteCarloRandomWalk(float *final_positions, int steps, float step_size,
                                   int walks_per_thread, unsigned int seed,
                                   int blockSize, int gridSize);
    void launchMonteCarloAreaEstimation(unsigned int *inside_count, float *shape_params,
                                       int samples_per_thread, unsigned int seed,
                                       int blockSize, int gridSize);
    
    double calculatePiEstimate(const unsigned int *inside_counts, int num_threads, int samples_per_thread);
    double calculateIntegralEstimate(const float *results, int num_threads);
    double calculateOptionPrice(const float *results, int num_threads);
    double calculateAverageDisplacement(const float *final_positions, int num_threads);
    double calculateAreaEstimate(const unsigned int *inside_counts, int num_threads, int samples_per_thread);
}

class MonteCarloSimulation
{
private:
    int num_threads;
    int samples_per_thread;
    int blockSize;
    int gridSize;
    
public:
    MonteCarloSimulation(int threads, int samples) : num_threads(threads), samples_per_thread(samples)
    {
        blockSize = 256;
        gridSize = (num_threads + blockSize - 1) / blockSize;
    }
    
    void runPiEstimation()
    {
        std::cout << "\n=== Monte Carlo Pi Estimation ===" << std::endl;
        
        // Allocate memory
        unsigned int *h_results = new unsigned int[num_threads];
        unsigned int *d_results;
        HIP_CHECK(hipMalloc(&d_results, num_threads * sizeof(unsigned int)));
        
        // Generate random seed
        std::random_device rd;
        unsigned int seed = rd();
        
        // Create events for timing
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        // Warmup
        launchMonteCarloPi(d_results, samples_per_thread, seed, blockSize, gridSize);
        
        // Benchmark
        HIP_CHECK(hipEventRecord(start));
        launchMonteCarloPi(d_results, samples_per_thread, seed, blockSize, gridSize);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));
        
        // Copy results back
        HIP_CHECK(hipMemcpy(h_results, d_results, num_threads * sizeof(unsigned int), hipMemcpyDeviceToHost));
        
        // Calculate Pi estimate
        double pi_estimate = calculatePiEstimate(h_results, num_threads, samples_per_thread);
        double error = fabs(pi_estimate - M_PI);
        
        long long total_samples = (long long)num_threads * samples_per_thread;
        double samples_per_second = total_samples / (gpu_time / 1000.0);
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Total samples: " << total_samples << std::endl;
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Samples/second: " << samples_per_second / 1e6 << " million" << std::endl;
        std::cout << "Pi estimate: " << pi_estimate << std::endl;
        std::cout << "Actual Pi: " << M_PI << std::endl;
        std::cout << "Error: " << error << std::endl;
        std::cout << "Relative error: " << (error / M_PI) * 100.0 << "%" << std::endl;
        
        // Cleanup
        delete[] h_results;
        HIP_CHECK(hipFree(d_results));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
    
    void runIntegrationExample()
    {
        std::cout << "\n=== Monte Carlo Integration (∫₀¹ x² dx) ===" << std::endl;
        
        // Allocate memory
        float *h_results = new float[num_threads];
        float *d_results;
        HIP_CHECK(hipMalloc(&d_results, num_threads * sizeof(float)));
        
        // Generate random seed
        std::random_device rd;
        unsigned int seed = rd();
        
        // Create events for timing
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        // Benchmark
        HIP_CHECK(hipEventRecord(start));
        launchMonteCarloIntegration(d_results, samples_per_thread, seed, blockSize, gridSize);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));
        
        // Copy results back
        HIP_CHECK(hipMemcpy(h_results, d_results, num_threads * sizeof(float), hipMemcpyDeviceToHost));
        
        // Calculate integral estimate
        double integral_estimate = calculateIntegralEstimate(h_results, num_threads);
        double analytical_result = 1.0 / 3.0; // ∫₀¹ x² dx = 1/3
        double error = fabs(integral_estimate - analytical_result);
        
        long long total_samples = (long long)num_threads * samples_per_thread;
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Total samples: " << total_samples << std::endl;
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Integral estimate: " << integral_estimate << std::endl;
        std::cout << "Analytical result: " << analytical_result << std::endl;
        std::cout << "Error: " << error << std::endl;
        std::cout << "Relative error: " << (error / analytical_result) * 100.0 << "%" << std::endl;
        
        // Cleanup
        delete[] h_results;
        HIP_CHECK(hipFree(d_results));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
    
    void runOptionPricing()
    {
        std::cout << "\n=== Monte Carlo Option Pricing (Black-Scholes) ===" << std::endl;
        
        // Option parameters
        float S0 = 100.0f;    // Initial stock price
        float K = 105.0f;     // Strike price
        float T = 1.0f;       // Time to expiration (1 year)
        float r = 0.05f;      // Risk-free rate
        float sigma = 0.2f;   // Volatility
        
        // Allocate memory
        float *h_results = new float[num_threads];
        float *d_results;
        HIP_CHECK(hipMalloc(&d_results, num_threads * sizeof(float)));
        
        // Generate random seed
        std::random_device rd;
        unsigned int seed = rd();
        
        // Create events for timing
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        // Benchmark
        HIP_CHECK(hipEventRecord(start));
        launchMonteCarloOptionPricing(d_results, S0, K, T, r, sigma, samples_per_thread, seed,
                                     blockSize, gridSize);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));
        
        // Copy results back
        HIP_CHECK(hipMemcpy(h_results, d_results, num_threads * sizeof(float), hipMemcpyDeviceToHost));
        
        // Calculate option price
        double option_price = calculateOptionPrice(h_results, num_threads);
        
        long long total_samples = (long long)num_threads * samples_per_thread;
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Option parameters:" << std::endl;
        std::cout << "  S0 (initial price): $" << S0 << std::endl;
        std::cout << "  K (strike price): $" << K << std::endl;
        std::cout << "  T (time to expiry): " << T << " years" << std::endl;
        std::cout << "  r (risk-free rate): " << r * 100.0f << "%" << std::endl;
        std::cout << "  σ (volatility): " << sigma * 100.0f << "%" << std::endl;
        std::cout << "Total samples: " << total_samples << std::endl;
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Call option price: $" << option_price << std::endl;
        
        // Cleanup
        delete[] h_results;
        HIP_CHECK(hipFree(d_results));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
    
    void runRandomWalkSimulation()
    {
        std::cout << "\n=== Monte Carlo Random Walk Simulation ===" << std::endl;
        
        int steps = 1000;
        float step_size = 1.0f;
        int walks_per_thread = samples_per_thread / 10; // Fewer walks but more complex
        
        // Allocate memory
        float *h_results = new float[num_threads];
        float *d_results;
        HIP_CHECK(hipMalloc(&d_results, num_threads * sizeof(float)));
        
        // Generate random seed
        std::random_device rd;
        unsigned int seed = rd();
        
        // Create events for timing
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        // Benchmark
        HIP_CHECK(hipEventRecord(start));
        launchMonteCarloRandomWalk(d_results, steps, step_size, walks_per_thread, seed,
                                  blockSize, gridSize);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));
        
        // Copy results back
        HIP_CHECK(hipMemcpy(h_results, d_results, num_threads * sizeof(float), hipMemcpyDeviceToHost));
        
        // Calculate average displacement
        double avg_displacement = calculateAverageDisplacement(h_results, num_threads);
        
        // Theoretical result for 1D random walk: sqrt(N) * step_size
        double theoretical_displacement = sqrt(steps) * step_size;
        
        long long total_walks = (long long)num_threads * walks_per_thread;
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Walk parameters:" << std::endl;
        std::cout << "  Steps per walk: " << steps << std::endl;
        std::cout << "  Step size: " << step_size << std::endl;
        std::cout << "  Total walks: " << total_walks << std::endl;
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Average displacement: " << avg_displacement << std::endl;
        std::cout << "Theoretical (√N): " << theoretical_displacement << std::endl;
        std::cout << "Ratio: " << avg_displacement / theoretical_displacement << std::endl;
        
        // Cleanup
        delete[] h_results;
        HIP_CHECK(hipFree(d_results));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
};

int main(int argc, char **argv)
{
    std::cout << "=== Monte Carlo Methods GPU Benchmark ===" << std::endl;
    
    // Initialize HIP
    initializeHIP();
    printHIPDeviceInfo();
    
    // Parse command line arguments
    int num_threads = (argc > 1) ? std::stoi(argv[1]) : 65536;
    int samples_per_thread = (argc > 2) ? std::stoi(argv[2]) : 1024;
    
    std::cout << "\nSimulation Configuration:" << std::endl;
    std::cout << "Number of threads: " << num_threads << std::endl;
    std::cout << "Samples per thread: " << samples_per_thread << std::endl;
    std::cout << "Total samples: " << (long long)num_threads * samples_per_thread << std::endl;
    
    try
    {
        MonteCarloSimulation simulation(num_threads, samples_per_thread);
        
        // Run all Monte Carlo examples
        simulation.runPiEstimation();
        simulation.runIntegrationExample();
        simulation.runOptionPricing();
        simulation.runRandomWalkSimulation();
        
        std::cout << "\n=== Monte Carlo Benchmark Completed ===" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
