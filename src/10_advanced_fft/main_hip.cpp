#include <hip/hip_runtime.h>
#include "hip_utils.h"
#include "timer.h"
#include "helper_functions.h"
#include <iostream>
#include <vector>
#include <complex>
#include <iomanip>
#include <random>
#include <cmath>

// Forward declaration of kernel
__global__ void fft3D_complex(
    const float2 *__restrict__ input,
    float2 *__restrict__ output,
    int NX, int NY, int NZ,
    int direction);

class Advanced3DFFTBenchmark
{
public:
    void runFFT3D(int NX, int NY, int NZ)
    {
        std::cout << "\n=== Advanced 3D FFT with Complex Numbers ===" << std::endl;
        std::cout << "Volume size: " << NX << "x" << NY << "x" << NZ << std::endl;

        const int total_size = NX * NY * NZ;
        const size_t bytes = total_size * sizeof(float2);

        // Allocate memory
        float2 *d_input, *d_output;
        HIP_CHECK(hipMalloc(&d_input, bytes));
        HIP_CHECK(hipMalloc(&d_output, bytes));

        // Initialize input data with complex sinusoidal pattern
        std::vector<float2> h_input(total_size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (int z = 0; z < NZ; z++)
        {
            for (int y = 0; y < NY; y++)
            {
                for (int x = 0; x < NX; x++)
                {
                    int idx = z * NY * NX + y * NX + x;

                    // Create complex pattern with multiple frequencies
                    float real_part = cos(2.0f * M_PI * x / NX) *
                                      cos(2.0f * M_PI * y / NY) *
                                      cos(2.0f * M_PI * z / NZ);
                    float imag_part = sin(2.0f * M_PI * x / NX) *
                                      sin(2.0f * M_PI * y / NY) *
                                      sin(2.0f * M_PI * z / NZ);

                    h_input[idx] = make_float2(real_part + dis(gen) * 0.1f,
                                               imag_part + dis(gen) * 0.1f);
                }
            }
        }

        HIP_CHECK(hipMemcpy(d_input, h_input.data(), bytes, hipMemcpyHostToDevice));

        // Launch configuration
        dim3 blockSize(8, 8, 8);
        dim3 gridSize((NX + blockSize.x - 1) / blockSize.x,
                      (NY + blockSize.y - 1) / blockSize.y,
                      (NZ + blockSize.z - 1) / blockSize.z);

        // Benchmark forward FFT
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        const int iterations = 10;

        std::cout << "Running forward FFT..." << std::endl;
        HIP_CHECK(hipEventRecord(start));

        for (int i = 0; i < iterations; i++)
        {
            hipLaunchKernelGGL(fft3D_complex, gridSize, blockSize, 0, 0,
                               d_input, d_output, NX, NY, NZ, 1); // Forward
        }

        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float fft_time;
        HIP_CHECK(hipEventElapsedTime(&fft_time, start, stop));
        fft_time /= iterations;

        // Benchmark inverse FFT
        std::cout << "Running inverse FFT..." << std::endl;
        HIP_CHECK(hipEventRecord(start));

        for (int i = 0; i < iterations; i++)
        {
            hipLaunchKernelGGL(fft3D_complex, gridSize, blockSize, 0, 0,
                               d_output, d_input, NX, NY, NZ, -1); // Inverse
        }

        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float ifft_time;
        HIP_CHECK(hipEventElapsedTime(&ifft_time, start, stop));
        ifft_time /= iterations;

        // Performance metrics
        double total_ops = total_size * 5.0 * log2(total_size); // Approximate FFT operations
        double gflops_fft = total_ops / (fft_time / 1000.0) / 1e9;
        double gflops_ifft = total_ops / (ifft_time / 1000.0) / 1e9;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n=== Performance Results ===" << std::endl;
        std::cout << "Forward FFT time: " << fft_time << " ms" << std::endl;
        std::cout << "Inverse FFT time: " << ifft_time << " ms" << std::endl;
        std::cout << "Forward FFT performance: " << gflops_fft << " GFLOPS" << std::endl;
        std::cout << "Inverse FFT performance: " << gflops_ifft << " GFLOPS" << std::endl;

        // Verify round-trip accuracy
        std::vector<float2> h_result(total_size);
        HIP_CHECK(hipMemcpy(h_result.data(), d_input, bytes, hipMemcpyDeviceToHost));

        double max_error = 0.0;
        double avg_error = 0.0;

        for (int i = 0; i < total_size; i++)
        {
            double error_real = fabs(h_input[i].x - h_result[i].x / total_size);
            double error_imag = fabs(h_input[i].y - h_result[i].y / total_size);
            double error = sqrt(error_real * error_real + error_imag * error_imag);

            max_error = std::max(max_error, error);
            avg_error += error;
        }
        avg_error /= total_size;

        std::cout << "\n=== Accuracy Results ===" << std::endl;
        std::cout << "Max round-trip error: " << max_error << std::endl;
        std::cout << "Average round-trip error: " << avg_error << std::endl;
        std::cout << "FFT Accuracy: " << (max_error < 1e-4 ? "GOOD" : "POOR") << std::endl;

        // Memory bandwidth
        double bytes_transferred = bytes * 2; // Read + Write
        double bandwidth_fft = bytes_transferred / (fft_time / 1000.0) / (1024 * 1024 * 1024);
        double bandwidth_ifft = bytes_transferred / (ifft_time / 1000.0) / (1024 * 1024 * 1024);

        std::cout << "\n=== Memory Performance ===" << std::endl;
        std::cout << "Forward FFT bandwidth: " << bandwidth_fft << " GB/s" << std::endl;
        std::cout << "Inverse FFT bandwidth: " << bandwidth_ifft << " GB/s" << std::endl;

        // Cleanup
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    void runFFTSweep()
    {
        std::cout << "\n=== FFT Size Performance Sweep ===" << std::endl;

        std::vector<int> sizes = {32, 64, 128, 256};

        for (int size : sizes)
        {
            std::cout << "\n--- Testing " << size << "^3 FFT ---" << std::endl;
            runFFT3D(size, size, size);
        }
    }
};

int main(int argc, char **argv)
{
    std::cout << "=== Advanced 3D FFT with Complex Numbers Benchmark ===" << std::endl;

    // Initialize HIP
    initializeHIP();
    printHIPDeviceInfo();

    // Parse command line arguments
    int size = (argc > 1) ? std::stoi(argv[1]) : 64;
    bool sweep = (argc > 2) && (std::string(argv[2]) == "sweep");

    try
    {
        Advanced3DFFTBenchmark benchmark;

        if (sweep)
        {
            benchmark.runFFTSweep();
        }
        else
        {
            std::cout << "\nFFT size: " << size << "^3" << std::endl;
            benchmark.runFFT3D(size, size, size);
        }

        std::cout << "\n=== Advanced 3D FFT Benchmark Completed Successfully ===" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
