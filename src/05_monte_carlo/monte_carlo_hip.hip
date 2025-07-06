#include "hip_utils.h"
#include <hip/hip_runtime.h>

// Simple Linear Congruential Generator for random numbers
__device__ float simple_random(unsigned int *state)
{
    *state = (*state * 1103515245 + 12345) & 0x7fffffff;
    return (float)(*state) / (float)0x7fffffff;
}

// Monte Carlo Pi estimation kernel
__global__ void monte_carlo_pi(unsigned int *results, int samples_per_thread, unsigned int seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize random state for this thread
    unsigned int rand_state = seed + tid;
    
    int inside_circle = 0;
    
    for (int i = 0; i < samples_per_thread; i++)
    {
        // Generate random point in unit square
        float x = simple_random(&rand_state);
        float y = simple_random(&rand_state);
        
        // Check if point is inside unit circle
        if (x * x + y * y <= 1.0f)
        {
            inside_circle++;
        }
    }
    
    results[tid] = inside_circle;
}

// Monte Carlo integration kernel (approximate integral of f(x) = x^2 from 0 to 1)
__global__ void monte_carlo_integration(float *results, int samples_per_thread, unsigned int seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned int rand_state = seed + tid;
    
    float sum = 0.0f;
    
    for (int i = 0; i < samples_per_thread; i++)
    {
        float x = simple_random(&rand_state);
        float fx = x * x; // f(x) = x^2
        sum += fx;
    }
    
    results[tid] = sum / samples_per_thread; // Average value
}

// Monte Carlo option pricing (Black-Scholes)
__global__ void monte_carlo_option_pricing(float *results, float S0, float K, float T, 
                                          float r, float sigma, int samples_per_thread, 
                                          unsigned int seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned int rand_state = seed + tid;
    
    float payoff_sum = 0.0f;
    
    for (int i = 0; i < samples_per_thread; i++)
    {
        // Generate random normal variable using Box-Muller transform
        float u1 = simple_random(&rand_state);
        float u2 = simple_random(&rand_state);
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
        
        // Calculate stock price at expiration
        float ST = S0 * expf((r - 0.5f * sigma * sigma) * T + sigma * sqrtf(T) * z);
        
        // Calculate call option payoff
        float payoff = fmaxf(ST - K, 0.0f);
        payoff_sum += payoff;
    }
    
    // Discounted expected payoff
    results[tid] = expf(-r * T) * payoff_sum / samples_per_thread;
}

// Monte Carlo random walk simulation
__global__ void monte_carlo_random_walk(float *final_positions, int steps, float step_size,
                                       int walks_per_thread, unsigned int seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned int rand_state = seed + tid;
    
    float total_displacement = 0.0f;
    
    for (int walk = 0; walk < walks_per_thread; walk++)
    {
        float position = 0.0f;
        
        for (int step = 0; step < steps; step++)
        {
            // Random step: +1 or -1
            float random = simple_random(&rand_state);
            float direction = (random < 0.5f) ? -1.0f : 1.0f;
            position += direction * step_size;
        }
        
        total_displacement += fabsf(position);
    }
    
    final_positions[tid] = total_displacement / walks_per_thread;
}

// Monte Carlo area estimation for complex shapes
__global__ void monte_carlo_area_estimation(unsigned int *inside_count, float *shape_params,
                                           int samples_per_thread, unsigned int seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned int rand_state = seed + tid;
    
    int inside = 0;
    
    // Parameters for a heart shape: ((x^2 + y^2 - 1)^3) <= x^2 * y^3
    for (int i = 0; i < samples_per_thread; i++)
    {
        // Generate random point in bounding box [-2, 2] x [-1.5, 1.5]
        float x = simple_random(&rand_state) * 4.0f - 2.0f;
        float y = simple_random(&rand_state) * 3.0f - 1.5f;
        
        // Heart shape equation
        float left_side = powf(x*x + y*y - 1.0f, 3.0f);
        float right_side = x*x * y*y*y;
        
        if (left_side <= right_side)
        {
            inside++;
        }
    }
    
    inside_count[tid] = inside;
}

// Host function wrappers
extern "C" {

void launchMonteCarloPi(unsigned int *results, int samples_per_thread, unsigned int seed,
                       int blockSize, int gridSize)
{
    hipLaunchKernelGGL(monte_carlo_pi, dim3(gridSize), dim3(blockSize), 0, 0,
                       results, samples_per_thread, seed);
    HIP_CHECK(hipDeviceSynchronize());
}

void launchMonteCarloIntegration(float *results, int samples_per_thread, unsigned int seed,
                                int blockSize, int gridSize)
{
    hipLaunchKernelGGL(monte_carlo_integration, dim3(gridSize), dim3(blockSize), 0, 0,
                       results, samples_per_thread, seed);
    HIP_CHECK(hipDeviceSynchronize());
}

void launchMonteCarloOptionPricing(float *results, float S0, float K, float T, float r, float sigma,
                                  int samples_per_thread, unsigned int seed,
                                  int blockSize, int gridSize)
{
    hipLaunchKernelGGL(monte_carlo_option_pricing, dim3(gridSize), dim3(blockSize), 0, 0,
                       results, S0, K, T, r, sigma, samples_per_thread, seed);
    HIP_CHECK(hipDeviceSynchronize());
}

void launchMonteCarloRandomWalk(float *final_positions, int steps, float step_size,
                               int walks_per_thread, unsigned int seed,
                               int blockSize, int gridSize)
{
    hipLaunchKernelGGL(monte_carlo_random_walk, dim3(gridSize), dim3(blockSize), 0, 0,
                       final_positions, steps, step_size, walks_per_thread, seed);
    HIP_CHECK(hipDeviceSynchronize());
}

void launchMonteCarloAreaEstimation(unsigned int *inside_count, float *shape_params,
                                   int samples_per_thread, unsigned int seed,
                                   int blockSize, int gridSize)
{
    hipLaunchKernelGGL(monte_carlo_area_estimation, dim3(gridSize), dim3(blockSize), 0, 0,
                       inside_count, shape_params, samples_per_thread, seed);
    HIP_CHECK(hipDeviceSynchronize());
}

}

// Utility functions
double calculatePiEstimate(const unsigned int *inside_counts, int num_threads, int samples_per_thread)
{
    unsigned long long total_inside = 0;
    for (int i = 0; i < num_threads; i++)
    {
        total_inside += inside_counts[i];
    }
    
    unsigned long long total_samples = (unsigned long long)num_threads * samples_per_thread;
    return 4.0 * (double)total_inside / (double)total_samples;
}

double calculateIntegralEstimate(const float *results, int num_threads)
{
    double sum = 0.0;
    for (int i = 0; i < num_threads; i++)
    {
        sum += results[i];
    }
    return sum / num_threads; // This approximates the integral of x^2 from 0 to 1 (should be ~0.333)
}

double calculateOptionPrice(const float *results, int num_threads)
{
    double sum = 0.0;
    for (int i = 0; i < num_threads; i++)
    {
        sum += results[i];
    }
    return sum / num_threads;
}

double calculateAverageDisplacement(const float *final_positions, int num_threads)
{
    double sum = 0.0;
    for (int i = 0; i < num_threads; i++)
    {
        sum += final_positions[i];
    }
    return sum / num_threads;
}

double calculateAreaEstimate(const unsigned int *inside_counts, int num_threads, int samples_per_thread)
{
    unsigned long long total_inside = 0;
    for (int i = 0; i < num_threads; i++)
    {
        total_inside += inside_counts[i];
    }
    
    unsigned long long total_samples = (unsigned long long)num_threads * samples_per_thread;
    double ratio = (double)total_inside / (double)total_samples;
    
    // Bounding box area is 4.0 * 3.0 = 12.0
    return ratio * 12.0;
}
