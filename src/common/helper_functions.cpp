#include "helper_functions.h"
#include <fstream>
#include <sstream>
#include <iomanip>

std::string formatBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
    return oss.str();
}

std::string formatTime(double milliseconds) {
    if (milliseconds < 1.0) {
        return std::to_string(milliseconds * 1000.0) + " μs";
    } else if (milliseconds < 1000.0) {
        return std::to_string(milliseconds) + " ms";
    } else {
        return std::to_string(milliseconds / 1000.0) + " s";
    }
}

std::string formatNumber(double number) {
    std::ostringstream oss;
    oss.imbue(std::locale(""));
    oss << std::fixed << std::setprecision(0) << number;
    return oss.str();
}

bool saveArrayToFile(const float* array, size_t size, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    file.write(reinterpret_cast<const char*>(array), size * sizeof(float));
    return file.good();
}

bool loadArrayFromFile(float* array, size_t size, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return false;
    }
    
    file.read(reinterpret_cast<char*>(array), size * sizeof(float));
    return file.good();
}

size_t alignUp(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

size_t alignDown(size_t value, size_t alignment) {
    return (value / alignment) * alignment;
}

void reportError(const std::string& message, const std::string& file, int line) {
    std::cerr << "ERROR: " << message << std::endl;
    std::cerr << "  File: " << file << std::endl;
    std::cerr << "  Line: " << line << std::endl;
}

// Progress Bar Implementation
ProgressBar::ProgressBar(size_t total, const std::string& description)
    : total_(total), description_(description), last_printed_(0) {
    if (!description_.empty()) {
        std::cout << description_ << std::endl;
    }
    update(0);
}

void ProgressBar::update(size_t current) {
    size_t progress = (current * 100) / total_;
    
    if (progress != last_printed_) {
        size_t bar_progress = (current * BAR_WIDTH) / total_;
        
        std::cout << "\r[";
        for (size_t i = 0; i < BAR_WIDTH; ++i) {
            if (i < bar_progress) {
                std::cout << "=";
            } else if (i == bar_progress) {
                std::cout << ">";
            } else {
                std::cout << " ";
            }
        }
        std::cout << "] " << progress << "% (" << current << "/" << total_ << ")";
        std::cout.flush();
        
        last_printed_ = progress;
    }
}

void ProgressBar::finish() {
    update(total_);
    std::cout << std::endl;
}
