#include <hip/hip_runtime.h>
#include "hip_utils.h"
#include "timer.h"
#include "helper_functions.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <cmath>
#include <fstream>

// Body structure for N-body simulation
struct Body
{
    float4 position; // x, y, z, mass
    float4 velocity; // vx, vy, vz, unused
};

// Forward declaration of kernels
__global__ void nbody_kernel_shared(
    Body *bodies,
    float dt,
    float softening,
    int n);

__global__ void nbody_kernel_tiled(
    Body *bodies,
    float dt,
    float softening,
    int n);

class NBodySimulationBenchmark
{
private:
    std::mt19937 gen;
    std::uniform_real_distribution<float> pos_dist;
    std::uniform_real_distribution<float> vel_dist;
    std::uniform_real_distribution<float> mass_dist;

public:
    NBodySimulationBenchmark() : gen(std::random_device{}()),
                                 pos_dist(-10.0f, 10.0f),
                                 vel_dist(-1.0f, 1.0f),
                                 mass_dist(0.1f, 2.0f) {}

    void initializeBodies(std::vector<Body> &bodies, int n)
    {
        bodies.resize(n);

        for (int i = 0; i < n; i++)
        {
            // Initialize in a rough spherical distribution
            float theta = pos_dist(gen) * M_PI;
            float phi = pos_dist(gen) * 2.0f * M_PI;
            float radius = abs(pos_dist(gen));

            bodies[i].position.x = radius * sin(theta) * cos(phi);
            bodies[i].position.y = radius * sin(theta) * sin(phi);
            bodies[i].position.z = radius * cos(theta);
            bodies[i].position.w = mass_dist(gen); // mass

            // Initial velocity (slight rotation)
            bodies[i].velocity.x = vel_dist(gen) * 0.1f;
            bodies[i].velocity.y = vel_dist(gen) * 0.1f;
            bodies[i].velocity.z = vel_dist(gen) * 0.1f;
            bodies[i].velocity.w = 0.0f;
        }
    }

    double calculateSystemEnergy(const std::vector<Body> &bodies)
    {
        double kinetic = 0.0, potential = 0.0;
        int n = bodies.size();

        for (int i = 0; i < n; i++)
        {
            // Kinetic energy
            double v2 = bodies[i].velocity.x * bodies[i].velocity.x +
                        bodies[i].velocity.y * bodies[i].velocity.y +
                        bodies[i].velocity.z * bodies[i].velocity.z;
            kinetic += 0.5 * bodies[i].position.w * v2;

            // Potential energy
            for (int j = i + 1; j < n; j++)
            {
                double dx = bodies[i].position.x - bodies[j].position.x;
                double dy = bodies[i].position.y - bodies[j].position.y;
                double dz = bodies[i].position.z - bodies[j].position.z;
                double r = sqrt(dx * dx + dy * dy + dz * dz + 1e-10);

                potential -= bodies[i].position.w * bodies[j].position.w / r;
            }
        }

        return kinetic + potential;
    }

    void runNBodySimulation(int n_bodies, int n_steps, bool use_tiled = false)
    {
        std::cout << "\n=== N-Body Simulation with " << (use_tiled ? "Tiled" : "Shared Memory")
                  << " Optimization ===" << std::endl;
        std::cout << "Bodies: " << n_bodies << ", Steps: " << n_steps << std::endl;

        // Initialize bodies
        std::vector<Body> h_bodies;
        initializeBodies(h_bodies, n_bodies);

        double initial_energy = calculateSystemEnergy(h_bodies);
        std::cout << "Initial system energy: " << std::fixed << std::setprecision(6)
                  << initial_energy << std::endl;

        // Allocate GPU memory
        Body *d_bodies;
        size_t bytes = n_bodies * sizeof(Body);
        HIP_CHECK(hipMalloc(&d_bodies, bytes));
        HIP_CHECK(hipMemcpy(d_bodies, h_bodies.data(), bytes, hipMemcpyHostToDevice));

        // Simulation parameters
        float dt = 0.01f;
        float softening = 0.1f;

        // Launch configuration
        int blockSize = 256;
        int gridSize = (n_bodies + blockSize - 1) / blockSize;

        // Timing
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        std::cout << "Running simulation..." << std::endl;
        HIP_CHECK(hipEventRecord(start));

        // Main simulation loop
        for (int step = 0; step < n_steps; step++)
        {
            if (use_tiled)
            {
                hipLaunchKernelGGL(nbody_kernel_tiled, dim3(gridSize), dim3(blockSize),
                                   blockSize * sizeof(Body), 0,
                                   d_bodies, dt, softening, n_bodies);
            }
            else
            {
                hipLaunchKernelGGL(nbody_kernel_shared, dim3(gridSize), dim3(blockSize),
                                   blockSize * sizeof(Body), 0,
                                   d_bodies, dt, softening, n_bodies);
            }

            // Check for errors periodically
            if (step % 100 == 0)
            {
                HIP_CHECK(hipDeviceSynchronize());
            }
        }

        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float simulation_time;
        HIP_CHECK(hipEventElapsedTime(&simulation_time, start, stop));

        // Copy results back
        HIP_CHECK(hipMemcpy(h_bodies.data(), d_bodies, bytes, hipMemcpyDeviceToHost));

        double final_energy = calculateSystemEnergy(h_bodies);
        double energy_drift = abs((final_energy - initial_energy) / initial_energy) * 100.0;

        // Performance metrics
        long long total_interactions = (long long)n_bodies * n_bodies * n_steps;
        double interactions_per_second = total_interactions / (simulation_time / 1000.0);
        double gflops = total_interactions * 20 / (simulation_time / 1000.0) / 1e9; // ~20 FLOPs per interaction

        std::cout << "\n=== Simulation Results ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Total time: " << simulation_time << " ms" << std::endl;
        std::cout << "Time per step: " << simulation_time / n_steps << " ms" << std::endl;
        std::cout << "Interactions/second: " << std::scientific << interactions_per_second << std::endl;
        std::cout << "Performance: " << std::fixed << gflops << " GFLOPS" << std::endl;

        std::cout << "\n=== Energy Conservation ===" << std::endl;
        std::cout << "Initial energy: " << std::setprecision(6) << initial_energy << std::endl;
        std::cout << "Final energy: " << final_energy << std::endl;
        std::cout << "Energy drift: " << std::setprecision(3) << energy_drift << "%" << std::endl;
        std::cout << "Conservation: " << (energy_drift < 5.0 ? "GOOD" : "POOR") << std::endl;

        // Memory bandwidth
        double bytes_per_step = n_bodies * sizeof(Body) * 2; // Read + Write
        double total_bytes = bytes_per_step * n_steps;
        double bandwidth = total_bytes / (simulation_time / 1000.0) / (1024 * 1024 * 1024);

        std::cout << "\n=== Memory Performance ===" << std::endl;
        std::cout << "Memory bandwidth: " << bandwidth << " GB/s" << std::endl;

        // Cleanup
        HIP_CHECK(hipFree(d_bodies));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    void runScalingTest()
    {
        std::cout << "\n=== N-Body Scaling Performance Test ===" << std::endl;

        std::vector<int> body_counts = {512, 1024, 2048, 4096, 8192};
        const int steps = 100;

        std::cout << "\nBodyCount\tShared(ms)\tTiled(ms)\tShared(GFLOPS)\tTiled(GFLOPS)" << std::endl;
        std::cout << "-----------------------------------------------------------------------" << std::endl;

        for (int n : body_counts)
        {
            // Test shared memory version
            std::vector<Body> h_bodies;
            initializeBodies(h_bodies, n);

            Body *d_bodies;
            size_t bytes = n * sizeof(Body);
            HIP_CHECK(hipMalloc(&d_bodies, bytes));
            HIP_CHECK(hipMemcpy(d_bodies, h_bodies.data(), bytes, hipMemcpyHostToDevice));

            float dt = 0.01f, softening = 0.1f;
            int blockSize = 256, gridSize = (n + blockSize - 1) / blockSize;

            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            // Shared memory version
            HIP_CHECK(hipEventRecord(start));
            for (int step = 0; step < steps; step++)
            {
                hipLaunchKernelGGL(nbody_kernel_shared, dim3(gridSize), dim3(blockSize),
                                   blockSize * sizeof(Body), 0,
                                   d_bodies, dt, softening, n);
            }
            HIP_CHECK(hipEventRecord(stop));
            HIP_CHECK(hipEventSynchronize(stop));

            float shared_time;
            HIP_CHECK(hipEventElapsedTime(&shared_time, start, stop));

            // Tiled version
            HIP_CHECK(hipMemcpy(d_bodies, h_bodies.data(), bytes, hipMemcpyHostToDevice));
            HIP_CHECK(hipEventRecord(start));
            for (int step = 0; step < steps; step++)
            {
                hipLaunchKernelGGL(nbody_kernel_tiled, dim3(gridSize), dim3(blockSize),
                                   blockSize * sizeof(Body), 0,
                                   d_bodies, dt, softening, n);
            }
            HIP_CHECK(hipEventRecord(stop));
            HIP_CHECK(hipEventSynchronize(stop));

            float tiled_time;
            HIP_CHECK(hipEventElapsedTime(&tiled_time, start, stop));

            // Calculate GFLOPS
            long long total_ops = (long long)n * n * steps * 20;
            double shared_gflops = total_ops / (shared_time / 1000.0) / 1e9;
            double tiled_gflops = total_ops / (tiled_time / 1000.0) / 1e9;

            std::cout << n << "\t\t" << std::fixed << std::setprecision(1)
                      << shared_time << "\t\t" << tiled_time << "\t\t"
                      << shared_gflops << "\t\t" << tiled_gflops << std::endl;

            HIP_CHECK(hipFree(d_bodies));
            HIP_CHECK(hipEventDestroy(start));
            HIP_CHECK(hipEventDestroy(stop));
        }
    }
};

int main(int argc, char **argv)
{
    std::cout << "=== N-Body Simulation with Shared Memory Optimization ===" << std::endl;

    // Initialize HIP
    initializeHIP();
    printHIPDeviceInfo();

    // Parse command line arguments
    int n_bodies = (argc > 1) ? std::stoi(argv[1]) : 4096;
    int n_steps = (argc > 2) ? std::stoi(argv[2]) : 100;
    std::string mode = (argc > 3) ? argv[3] : "both";

    try
    {
        NBodySimulationBenchmark benchmark;

        if (mode == "scaling")
        {
            benchmark.runScalingTest();
        }
        else if (mode == "shared")
        {
            benchmark.runNBodySimulation(n_bodies, n_steps, false);
        }
        else if (mode == "tiled")
        {
            benchmark.runNBodySimulation(n_bodies, n_steps, true);
        }
        else
        {
            // Run both versions for comparison
            benchmark.runNBodySimulation(n_bodies, n_steps, false);
            benchmark.runNBodySimulation(n_bodies, n_steps, true);
        }

        std::cout << "\n=== N-Body Simulation Completed Successfully ===" << std::endl;
        std::cout << "\nUsage: " << argv[0] << " [n_bodies] [n_steps] [mode]" << std::endl;
        std::cout << "  n_bodies: Number of bodies (default: 4096)" << std::endl;
        std::cout << "  n_steps: Number of simulation steps (default: 100)" << std::endl;
        std::cout << "  mode: shared|tiled|both|scaling (default: both)" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
