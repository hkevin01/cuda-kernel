#include "hip_utils.h"
#include <hip/hip_runtime.h>

#define TILE_SIZE 16
#define FILTER_SIZE 5
#define FILTER_RADIUS (FILTER_SIZE / 2)

// Simple 2D convolution kernel (naive implementation)
__global__ void convolution2D_naive(float *input, float *filter, float *output,
                                   int width, int height, int filter_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width)
    {
        float sum = 0.0f;
        int filter_radius = filter_size / 2;
        
        for (int k_row = 0; k_row < filter_size; k_row++)
        {
            for (int k_col = 0; k_col < filter_size; k_col++)
            {
                int input_row = row + k_row - filter_radius;
                int input_col = col + k_col - filter_radius;
                
                // Handle boundary conditions (zero padding)
                if (input_row >= 0 && input_row < height && 
                    input_col >= 0 && input_col < width)
                {
                    sum += input[input_row * width + input_col] * 
                           filter[k_row * filter_size + k_col];
                }
            }
        }
        
        output[row * width + col] = sum;
    }
}

// Optimized 2D convolution with shared memory
__global__ void convolution2D_shared(float *input, float *filter, float *output,
                                    int width, int height, int filter_size)
{
    int filter_radius = filter_size / 2;
    
    // Shared memory for tile with halo region
    __shared__ float tile[TILE_SIZE + 2*FILTER_RADIUS][TILE_SIZE + 2*FILTER_RADIUS];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    
    // Load main tile data
    int shared_col = tx + filter_radius;
    int shared_row = ty + filter_radius;
    
    if (row < height && col < width)
    {
        tile[shared_row][shared_col] = input[row * width + col];
    }
    else
    {
        tile[shared_row][shared_col] = 0.0f;
    }
    
    // Load halo regions
    // Left halo
    if (tx < filter_radius)
    {
        int halo_col = col - filter_radius;
        if (halo_col >= 0 && row < height)
        {
            tile[shared_row][tx] = input[row * width + halo_col];
        }
        else
        {
            tile[shared_row][tx] = 0.0f;
        }
    }
    
    // Right halo
    if (tx >= blockDim.x - filter_radius)
    {
        int halo_col = col + filter_radius;
        if (halo_col < width && row < height)
        {
            tile[shared_row][tx + 2*filter_radius] = input[row * width + halo_col];
        }
        else
        {
            tile[shared_row][tx + 2*filter_radius] = 0.0f;
        }
    }
    
    // Top halo
    if (ty < filter_radius)
    {
        int halo_row = row - filter_radius;
        if (halo_row >= 0 && col < width)
        {
            tile[ty][shared_col] = input[halo_row * width + col];
        }
        else
        {
            tile[ty][shared_col] = 0.0f;
        }
    }
    
    // Bottom halo
    if (ty >= blockDim.y - filter_radius)
    {
        int halo_row = row + filter_radius;
        if (halo_row < height && col < width)
        {
            tile[ty + 2*filter_radius][shared_col] = input[halo_row * width + col];
        }
        else
        {
            tile[ty + 2*filter_radius][shared_col] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Perform convolution
    if (row < height && col < width)
    {
        float sum = 0.0f;
        
        for (int k_row = 0; k_row < filter_size; k_row++)
        {
            for (int k_col = 0; k_col < filter_size; k_col++)
            {
                sum += tile[ty + k_row][tx + k_col] * 
                       filter[k_row * filter_size + k_col];
            }
        }
        
        output[row * width + col] = sum;
    }
}

// Constant memory convolution (for small filters)
__constant__ float const_filter[FILTER_SIZE * FILTER_SIZE];

__global__ void convolution2D_constant(float *input, float *output,
                                      int width, int height, int filter_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width)
    {
        float sum = 0.0f;
        int filter_radius = filter_size / 2;
        
        for (int k_row = 0; k_row < filter_size; k_row++)
        {
            for (int k_col = 0; k_col < filter_size; k_col++)
            {
                int input_row = row + k_row - filter_radius;
                int input_col = col + k_col - filter_radius;
                
                // Handle boundary conditions
                if (input_row >= 0 && input_row < height && 
                    input_col >= 0 && input_col < width)
                {
                    sum += input[input_row * width + input_col] * 
                           const_filter[k_row * filter_size + k_col];
                }
            }
        }
        
        output[row * width + col] = sum;
    }
}

// Separable convolution (for separable filters)
__global__ void convolution1D_horizontal(float *input, float *output, float *filter,
                                        int width, int height, int filter_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width)
    {
        float sum = 0.0f;
        int filter_radius = filter_size / 2;
        
        for (int k = 0; k < filter_size; k++)
        {
            int input_col = col + k - filter_radius;
            
            if (input_col >= 0 && input_col < width)
            {
                sum += input[row * width + input_col] * filter[k];
            }
        }
        
        output[row * width + col] = sum;
    }
}

__global__ void convolution1D_vertical(float *input, float *output, float *filter,
                                      int width, int height, int filter_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width)
    {
        float sum = 0.0f;
        int filter_radius = filter_size / 2;
        
        for (int k = 0; k < filter_size; k++)
        {
            int input_row = row + k - filter_radius;
            
            if (input_row >= 0 && input_row < height)
            {
                sum += input[input_row * width + col] * filter[k];
            }
        }
        
        output[row * width + col] = sum;
    }
}

// Host function wrappers
extern "C" {

void launchConvolution2DNaive(float *input, float *filter, float *output,
                             int width, int height, int filter_size,
                             dim3 gridSize, dim3 blockSize)
{
    hipLaunchKernelGGL(convolution2D_naive, gridSize, blockSize, 0, 0,
                       input, filter, output, width, height, filter_size);
    HIP_CHECK(hipDeviceSynchronize());
}

void launchConvolution2DShared(float *input, float *filter, float *output,
                              int width, int height, int filter_size,
                              dim3 gridSize, dim3 blockSize)
{
    hipLaunchKernelGGL(convolution2D_shared, gridSize, blockSize, 0, 0,
                       input, filter, output, width, height, filter_size);
    HIP_CHECK(hipDeviceSynchronize());
}

void launchConvolution2DConstant(float *input, float *filter, float *output,
                                int width, int height, int filter_size,
                                dim3 gridSize, dim3 blockSize)
{
    // Copy filter to constant memory
    HIP_CHECK(hipMemcpyToSymbol(const_filter, filter, 
                               filter_size * filter_size * sizeof(float)));
    
    hipLaunchKernelGGL(convolution2D_constant, gridSize, blockSize, 0, 0,
                       input, output, width, height, filter_size);
    HIP_CHECK(hipDeviceSynchronize());
}

void launchSeparableConvolution(float *input, float *temp, float *output, float *filter,
                               int width, int height, int filter_size,
                               dim3 gridSize, dim3 blockSize)
{
    // Horizontal pass
    hipLaunchKernelGGL(convolution1D_horizontal, gridSize, blockSize, 0, 0,
                       input, temp, filter, width, height, filter_size);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Vertical pass
    hipLaunchKernelGGL(convolution1D_vertical, gridSize, blockSize, 0, 0,
                       temp, output, filter, width, height, filter_size);
    HIP_CHECK(hipDeviceSynchronize());
}

}

// Utility functions
void createGaussianFilter(float *filter, int size, float sigma)
{
    int radius = size / 2;
    float sum = 0.0f;
    
    // Generate Gaussian kernel
    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            int dx = x - radius;
            int dy = y - radius;
            float value = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
            filter[y * size + x] = value;
            sum += value;
        }
    }
    
    // Normalize
    for (int i = 0; i < size * size; i++)
    {
        filter[i] /= sum;
    }
}

void createSobelFilter(float *filter_x, float *filter_y)
{
    // Sobel X filter
    float sobel_x[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    
    // Sobel Y filter
    float sobel_y[9] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };
    
    for (int i = 0; i < 9; i++)
    {
        filter_x[i] = sobel_x[i];
        filter_y[i] = sobel_y[i];
    }
}

void initializeImage(float *image, int width, int height, bool random = false)
{
    if (random)
    {
        for (int i = 0; i < width * height; i++)
        {
            image[i] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    else
    {
        // Create a simple test pattern
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                image[y * width + x] = sinf(x * 0.1f) * cosf(y * 0.1f);
            }
        }
    }
}

void printImage(const float *image, int width, int height, const char *name)
{
    printf("\n%s (%dx%d):\n", name, width, height);
    int display_width = std::min(width, 16);
    int display_height = std::min(height, 16);
    
    for (int y = 0; y < display_height; y++)
    {
        for (int x = 0; x < display_width; x++)
        {
            printf("%6.2f ", image[y * width + x]);
        }
        if (width > 16) printf("...");
        printf("\n");
    }
    if (height > 16) printf("...\n");
}

// CPU reference implementation
void convolution2DCPU(const float *input, const float *filter, float *output,
                      int width, int height, int filter_size)
{
    int filter_radius = filter_size / 2;
    
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float sum = 0.0f;
            
            for (int ky = 0; ky < filter_size; ky++)
            {
                for (int kx = 0; kx < filter_size; kx++)
                {
                    int input_y = y + ky - filter_radius;
                    int input_x = x + kx - filter_radius;
                    
                    if (input_y >= 0 && input_y < height && 
                        input_x >= 0 && input_x < width)
                    {
                        sum += input[input_y * width + input_x] * 
                               filter[ky * filter_size + kx];
                    }
                }
            }
            
            output[y * width + x] = sum;
        }
    }
}

bool verifyConvolution(const float *gpu_result, const float *cpu_result,
                      int size, float tolerance = 1e-3f)
{
    for (int i = 0; i < size; i++)
    {
        if (fabsf(gpu_result[i] - cpu_result[i]) > tolerance)
        {
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n",
                   i, gpu_result[i], cpu_result[i]);
            return false;
        }
    }
    return true;
}
