#pragma once

// Matrix multiplication kernels
__global__ void matrixMulNaive(const float *A, const float *B, float *C,
                               int M, int N, int K);

__global__ void matrixMulShared(const float *A, const float *B, float *C,
                                int M, int N, int K);

__global__ void matrixMulOptimized(const float *A, const float *B, float *C,
                                   int M, int N, int K);

// CPU reference implementation
void matrixMulCPU(const float *A, const float *B, float *C,
                  int M, int N, int K);

// Utility functions
void initializeMatrix(float *matrix, int rows, int cols, float min_val = 0.0f, float max_val = 1.0f);
bool verifyMatrixResult(const float *C_cpu, const float *C_gpu, int M, int N, float tolerance = 1e-4f);
