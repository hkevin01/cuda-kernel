#pragma once

#include <chrono>
#include <iostream>
#include <string>

class CPUTimer {
public:
    CPUTimer() : running(false) {}
    
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }
    
    double stop() {
        if (!running) return 0.0;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        running = false;
        
        return duration.count() / 1000.0; // Return in milliseconds
    }
    
    double elapsed() const {
        if (!running) return 0.0;
        
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time);
        
        return duration.count() / 1000.0; // Return in milliseconds
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_time;
    bool running;
};

// Utility class for measuring and comparing performance
class PerformanceMetrics {
public:
    struct Results {
        double cpu_time_ms;
        double gpu_time_ms;
        double speedup;
        double bandwidth_gb_s;
        double gflops;
        bool verification_passed;
    };
    
    static void printResults(const std::string& test_name, const Results& results) {
        std::cout << "\n=== " << test_name << " Performance Results ===" << std::endl;
        std::cout << "CPU Time: " << results.cpu_time_ms << " ms" << std::endl;
        std::cout << "GPU Time: " << results.gpu_time_ms << " ms" << std::endl;
        std::cout << "Speedup: " << results.speedup << "x" << std::endl;
        std::cout << "Bandwidth: " << results.bandwidth_gb_s << " GB/s" << std::endl;
        std::cout << "Performance: " << results.gflops << " GFLOPS" << std::endl;
        std::cout << "Verification: " << (results.verification_passed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "================================================" << std::endl;
    }
};
