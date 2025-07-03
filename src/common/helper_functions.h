#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>

// Mathematical constants
const float PI = 3.14159265358979323846f;
const float E = 2.71828182845904523536f;

// Helper functions for array operations
template <typename T>
void printArray(const T *array, size_t size, const std::string &name, size_t max_elements = 10)
{
    std::cout << name << ": ";
    size_t elements_to_print = std::min(size, max_elements);

    for (size_t i = 0; i < elements_to_print; ++i)
    {
        std::cout << std::fixed << std::setprecision(4) << array[i];
        if (i < elements_to_print - 1)
            std::cout << ", ";
    }

    if (size > max_elements)
    {
        std::cout << " ... (showing " << elements_to_print << " of " << size << " elements)";
    }
    std::cout << std::endl;
}

template <typename T>
void fillArray(T *array, size_t size, T value)
{
    for (size_t i = 0; i < size; ++i)
    {
        array[i] = value;
    }
}

template <typename T>
T sumArray(const T *array, size_t size)
{
    T sum = T{0};
    for (size_t i = 0; i < size; ++i)
    {
        sum += array[i];
    }
    return sum;
}

template <typename T>
T maxArray(const T *array, size_t size)
{
    if (size == 0)
        return T{0};

    T max_val = array[0];
    for (size_t i = 1; i < size; ++i)
    {
        max_val = std::max(max_val, array[i]);
    }
    return max_val;
}

template <typename T>
T minArray(const T *array, size_t size)
{
    if (size == 0)
        return T{0};

    T min_val = array[0];
    for (size_t i = 1; i < size; ++i)
    {
        min_val = std::min(min_val, array[i]);
    }
    return min_val;
}

// Matrix operations
template <typename T>
void printMatrix(const T *matrix, int rows, int cols, const std::string &name)
{
    std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;

    int max_rows = std::min(rows, 8);
    int max_cols = std::min(cols, 8);

    for (int i = 0; i < max_rows; ++i)
    {
        for (int j = 0; j < max_cols; ++j)
        {
            std::cout << std::fixed << std::setprecision(4) << matrix[i * cols + j] << "\t";
        }
        if (cols > max_cols)
            std::cout << "...";
        std::cout << std::endl;
    }

    if (rows > max_rows)
    {
        std::cout << "..." << std::endl;
    }
    std::cout << std::endl;
}

// String utilities
std::string formatBytes(size_t bytes);
std::string formatTime(double milliseconds);
std::string formatNumber(double number);

// File I/O helpers
bool saveArrayToFile(const float *array, size_t size, const std::string &filename);
bool loadArrayFromFile(float *array, size_t size, const std::string &filename);

// Memory alignment helpers
template <typename T>
bool isAligned(const T *ptr, size_t alignment)
{
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

size_t alignUp(size_t value, size_t alignment);
size_t alignDown(size_t value, size_t alignment);

// Error handling
void reportError(const std::string &message, const std::string &file, int line);

#define REPORT_ERROR(msg) reportError(msg, __FILE__, __LINE__)

// Progress indicator
class ProgressBar
{
public:
    ProgressBar(size_t total, const std::string &description = "");
    void update(size_t current);
    void finish();

private:
    size_t total_;
    std::string description_;
    size_t last_printed_;
    static const size_t BAR_WIDTH = 50;
};
