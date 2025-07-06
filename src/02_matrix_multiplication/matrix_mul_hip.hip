#include "hip_utils.h"
#include <hip/hip_runtime.h>
#include <random>

// Block size for shared memory tiling
#define BLOCK_SIZE 16
#define TILE_SIZE 16

// Naive matrix multiplication kernel
__global__ void matrixMulNaive(const float *A, const float *B, float *C,
                               int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Matrix multiplication with shared memory tiling
__global__ void matrixMulShared(const float *A, const float *B, float *C,
                                int M, int N, int K)
{
    // Shared memory for tiles
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // Calculate output position
    int row = blockRow * BLOCK_SIZE + threadRow;
    int col = blockCol * BLOCK_SIZE + threadCol;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t)
    {
        // Load tile into shared memory
        int aRow = row;
        int aCol = t * BLOCK_SIZE + threadCol;
        int bRow = t * BLOCK_SIZE + threadRow;
        int bCol = col;

        if (aRow < M && aCol < K)
            As[threadRow][threadCol] = A[aRow * K + aCol];
        else
            As[threadRow][threadCol] = 0.0f;

        if (bRow < K && bCol < N)
            Bs[threadRow][threadCol] = B[bRow * N + bCol];
        else
            Bs[threadRow][threadCol] = 0.0f;

        __syncthreads();

        // Compute partial sum for this tile
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            sum += As[threadRow][k] * Bs[k][threadCol];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

// Matrix multiplication with memory coalescing optimization
__global__ void matrixMulCoalesced(const float *A, const float *B, float *C,
                                   int M, int N, int K)
{
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1]; // +1 to avoid bank conflicts
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int row = blockRow * BLOCK_SIZE + threadRow;
    int col = blockCol * BLOCK_SIZE + threadCol;

    float sum = 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t)
    {
        // Coalesced loading of tile A
        int aRow = row;
        int aCol = t * BLOCK_SIZE + threadCol;
        if (aRow < M && aCol < K)
            As[threadRow][threadCol] = A[aRow * K + aCol];
        else
            As[threadRow][threadCol] = 0.0f;

        // Coalesced loading of tile B
        int bRow = t * BLOCK_SIZE + threadRow;
        int bCol = col;
        if (bRow < K && bCol < N)
            Bs[threadRow][threadCol] = B[bRow * N + bCol];
        else
            Bs[threadRow][threadCol] = 0.0f;

        __syncthreads();

        // Compute using shared memory
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            sum += As[threadRow][k] * Bs[k][threadCol];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

// Host function wrappers
extern "C" {

void launchMatrixMulNaive(const float *A, const float *B, float *C,
                          int M, int N, int K,
                          dim3 gridSize, dim3 blockSize)
{
    hipLaunchKernelGGL(matrixMulNaive, gridSize, blockSize, 0, 0,
                       A, B, C, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());
}

void launchMatrixMulShared(const float *A, const float *B, float *C,
                           int M, int N, int K,
                           dim3 gridSize, dim3 blockSize)
{
    hipLaunchKernelGGL(matrixMulShared, gridSize, blockSize, 0, 0,
                       A, B, C, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());
}

void launchMatrixMulCoalesced(const float *A, const float *B, float *C,
                              int M, int N, int K,
                              dim3 gridSize, dim3 blockSize)
{
    hipLaunchKernelGGL(matrixMulCoalesced, gridSize, blockSize, 0, 0,
                       A, B, C, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());
}

}

// Matrix utility functions - wrapped for C++ linkage
extern "C" {

void initializeMatrix(float *matrix, int rows, int cols, bool random = true)
{
    if (random)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (int i = 0; i < rows * cols; i++)
        {
            matrix[i] = dis(gen);
        }
    }
    else
    {
        for (int i = 0; i < rows * cols; i++)
        {
            matrix[i] = 0.0f;
        }
    }
}

void printMatrix(const float *matrix, int rows, int cols, const char *name)
{
    printf("\n%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < std::min(rows, 8); i++)
    {
        for (int j = 0; j < std::min(cols, 8); j++)
        {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        if (cols > 8) printf("...");
        printf("\n");
    }
    if (rows > 8) printf("...\n");
}

// CPU reference implementation for verification
void matrixMulCPU(const float *A, const float *B, float *C,
                  int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Verification function
bool verifyResult(const float *gpuResult, const float *cpuResult,
                  int size, float tolerance = 1e-3f)
{
    for (int i = 0; i < size; i++)
    {
        if (abs(gpuResult[i] - cpuResult[i]) > tolerance)
        {
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n",
                   i, gpuResult[i], cpuResult[i]);
            return false;
        }
    }
    return true;
}

} // extern "C"
