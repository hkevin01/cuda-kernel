#include "matrix_mul.h"
#include "cuda_utils.h"
#include <random>

// Block size for shared memory tiling
#define BLOCK_SIZE 16
#define TILE_SIZE 16

#ifdef USE_CUDA
// CUDA-specific includes and kernel launches
#endif
// Use gpu_utils.h for platform-agnostic GPU code

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
    for (int tileIdx = 0; tileIdx < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tileIdx)
    {
        // Load tile into shared memory
        int aRow = row;
        int aCol = tileIdx * BLOCK_SIZE + threadCol;
        int bRow = tileIdx * BLOCK_SIZE + threadRow;
        int bCol = col;

        if (aRow < M && aCol < K)
        {
            As[threadRow][threadCol] = A[aRow * K + aCol];
        }
        else
        {
            As[threadRow][threadCol] = 0.0f;
        }

        if (bRow < K && bCol < N)
        {
            Bs[threadRow][threadCol] = B[bRow * N + bCol];
        }
        else
        {
            Bs[threadRow][threadCol] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum using shared memory
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

// Optimized matrix multiplication with multiple elements per thread
__global__ void matrixMulOptimized(const float *A, const float *B, float *C,
                                   int M, int N, int K)
{
    const int TILE_M = 4;
    const int TILE_N = 4;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // Each thread computes TILE_M x TILE_N elements
    float results[TILE_M][TILE_N] = {0.0f};

    // Loop over tiles in K dimension
    for (int tileIdx = 0; tileIdx < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tileIdx)
    {
        // Collaborative loading of shared memory
        for (int loadOffset = 0; loadOffset < TILE_M; ++loadOffset)
        {
            int aRow = blockRow * BLOCK_SIZE + threadRow + loadOffset * (BLOCK_SIZE / TILE_M);
            int aCol = tileIdx * BLOCK_SIZE + threadCol;

            if (aRow < M && aCol < K)
            {
                As[threadRow + loadOffset * (BLOCK_SIZE / TILE_M)][threadCol] = A[aRow * K + aCol];
            }
            else
            {
                As[threadRow + loadOffset * (BLOCK_SIZE / TILE_M)][threadCol] = 0.0f;
            }
        }

        for (int loadOffset = 0; loadOffset < TILE_N; ++loadOffset)
        {
            int bRow = tileIdx * BLOCK_SIZE + threadRow;
            int bCol = blockCol * BLOCK_SIZE + threadCol + loadOffset * (BLOCK_SIZE / TILE_N);

            if (bRow < K && bCol < N)
            {
                Bs[threadRow][threadCol + loadOffset * (BLOCK_SIZE / TILE_N)] = B[bRow * N + bCol];
            }
            else
            {
                Bs[threadRow][threadCol + loadOffset * (BLOCK_SIZE / TILE_N)] = 0.0f;
            }
        }

        __syncthreads();

        // Compute using shared memory
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            for (int m = 0; m < TILE_M; ++m)
            {
                for (int n = 0; n < TILE_N; ++n)
                {
                    int sharedRowA = threadRow + m * (BLOCK_SIZE / TILE_M);
                    int sharedColB = threadCol + n * (BLOCK_SIZE / TILE_N);

                    if (sharedRowA < BLOCK_SIZE && sharedColB < BLOCK_SIZE)
                    {
                        results[m][n] += As[sharedRowA][k] * Bs[k][sharedColB];
                    }
                }
            }
        }

        __syncthreads();
    }

    // Write results back to global memory
    for (int m = 0; m < TILE_M; ++m)
    {
        for (int n = 0; n < TILE_N; ++n)
        {
            int row = blockRow * BLOCK_SIZE + threadRow + m * (BLOCK_SIZE / TILE_M);
            int col = blockCol * BLOCK_SIZE + threadCol + n * (BLOCK_SIZE / TILE_N);

            if (row < M && col < N)
            {
                C[row * N + col] = results[m][n];
            }
        }
    }
}

// CPU reference implementation
void matrixMulCPU(const float *A, const float *B, float *C,
                  int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Initialize matrix with random values
void initializeMatrix(float *matrix, int rows, int cols, float min_val, float max_val)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);

    for (int i = 0; i < rows * cols; ++i)
    {
        matrix[i] = dist(gen);
    }
}

// Verify matrix multiplication result
bool verifyMatrixResult(const float *C_cpu, const float *C_gpu, int M, int N, float tolerance)
{
    for (int i = 0; i < M * N; ++i)
    {
        if (std::abs(C_cpu[i] - C_gpu[i]) > tolerance)
        {
            std::cerr << "Verification failed at index " << i
                      << ": CPU = " << C_cpu[i]
                      << ", GPU = " << C_gpu[i]
                      << ", diff = " << std::abs(C_cpu[i] - C_gpu[i]) << std::endl;
            return false;
        }
    }
    return true;
}
