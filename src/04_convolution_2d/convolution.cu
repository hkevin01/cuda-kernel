#include "convolution.h"
#include "../common/cuda_utils.h"
#include <cmath>

// Naive convolution kernel - simple but inefficient
__global__ void convolution_naive(const float *input, const float *kernel,
                                  float *output, int width, int height,
                                  int kernel_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        float sum = 0.0f;
        int kernel_radius = kernel_size / 2;

        for (int k_row = 0; k_row < kernel_size; k_row++)
        {
            for (int k_col = 0; k_col < kernel_size; k_col++)
            {
                int img_row = row + k_row - kernel_radius;
                int img_col = col + k_col - kernel_radius;

                // Handle boundary conditions (zero padding)
                if (img_row >= 0 && img_row < height &&
                    img_col >= 0 && img_col < width)
                {
                    sum += input[img_row * width + img_col] *
                           kernel[k_row * kernel_size + k_col];
                }
            }
        }

        output[row * width + col] = sum;
    }
}

// Shared memory convolution - cache input data in shared memory
__global__ void convolution_shared_memory(const float *input, const float *kernel,
                                          float *output, int width, int height,
                                          int kernel_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int kernel_radius = kernel_size / 2;
    int tile_size = blockDim.x;
    int shared_size = tile_size + 2 * kernel_radius;

    extern __shared__ float tile[];

    // Load data into shared memory with halo
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Each thread loads multiple elements to fill the shared memory tile
    for (int dy = ty; dy < shared_size; dy += blockDim.y)
    {
        for (int dx = tx; dx < shared_size; dx += blockDim.x)
        {
            int img_row = blockIdx.y * blockDim.y + dy - kernel_radius;
            int img_col = blockIdx.x * blockDim.x + dx - kernel_radius;

            // Handle boundary conditions
            if (img_row >= 0 && img_row < height &&
                img_col >= 0 && img_col < width)
            {
                tile[dy * shared_size + dx] = input[img_row * width + img_col];
            }
            else
            {
                tile[dy * shared_size + dx] = 0.0f;
            }
        }
    }

    __syncthreads();

    // Perform convolution using shared memory
    if (col < width && row < height)
    {
        float sum = 0.0f;
        int tile_row = ty + kernel_radius;
        int tile_col = tx + kernel_radius;

        for (int k_row = 0; k_row < kernel_size; k_row++)
        {
            for (int k_col = 0; k_col < kernel_size; k_col++)
            {
                sum += tile[(tile_row + k_row - kernel_radius) * shared_size +
                            (tile_col + k_col - kernel_radius)] *
                       kernel[k_row * kernel_size + k_col];
            }
        }

        output[row * width + col] = sum;
    }
}

// Constant memory convolution - kernel stored in constant memory
__global__ void convolution_constant_memory(const float *input, float *output,
                                            int width, int height, int kernel_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        float sum = 0.0f;
        int kernel_radius = kernel_size / 2;

        for (int k_row = 0; k_row < kernel_size; k_row++)
        {
            for (int k_col = 0; k_col < kernel_size; k_col++)
            {
                int img_row = row + k_row - kernel_radius;
                int img_col = col + k_col - kernel_radius;

                if (img_row >= 0 && img_row < height &&
                    img_col >= 0 && img_col < width)
                {
                    sum += input[img_row * width + img_col] *
                           d_kernel[k_row * kernel_size + k_col];
                }
            }
        }

        output[row * width + col] = sum;
    }
}

// Separable convolution - horizontal pass
__global__ void convolution_separable_horizontal(const float *input, const float *kernel_x,
                                                 float *temp, int width, int height,
                                                 int kernel_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        float sum = 0.0f;
        int kernel_radius = kernel_size / 2;

        for (int k = 0; k < kernel_size; k++)
        {
            int img_col = col + k - kernel_radius;
            if (img_col >= 0 && img_col < width)
            {
                sum += input[row * width + img_col] * kernel_x[k];
            }
        }

        temp[row * width + col] = sum;
    }
}

// Separable convolution - vertical pass
__global__ void convolution_separable_vertical(const float *temp, const float *kernel_y,
                                               float *output, int width, int height,
                                               int kernel_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        float sum = 0.0f;
        int kernel_radius = kernel_size / 2;

        for (int k = 0; k < kernel_size; k++)
        {
            int img_row = row + k - kernel_radius;
            if (img_row >= 0 && img_row < height)
            {
                sum += temp[img_row * width + col] * kernel_y[k];
            }
        }

        output[row * width + col] = sum;
    }
}

// Texture memory convolution
__global__ void convolution_texture(cudaTextureObject_t texObj, const float *kernel,
                                    float *output, int width, int height,
                                    int kernel_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        float sum = 0.0f;
        int kernel_radius = kernel_size / 2;

        for (int k_row = 0; k_row < kernel_size; k_row++)
        {
            for (int k_col = 0; k_col < kernel_size; k_col++)
            {
                float img_row = row + k_row - kernel_radius + 0.5f;
                float img_col = col + k_col - kernel_radius + 0.5f;

                // tex2D handles boundary conditions automatically
                sum += tex2D<float>(texObj, img_col, img_row) *
                       kernel[k_row * kernel_size + k_col];
            }
        }

        output[row * width + col] = sum;
    }
}

// CPU reference implementation
void convolution_cpu(const float *input, const float *kernel, float *output,
                     int width, int height, int kernel_size)
{
    int kernel_radius = kernel_size / 2;

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            float sum = 0.0f;

            for (int k_row = 0; k_row < kernel_size; k_row++)
            {
                for (int k_col = 0; k_col < kernel_size; k_col++)
                {
                    int img_row = row + k_row - kernel_radius;
                    int img_col = col + k_col - kernel_radius;

                    if (img_row >= 0 && img_row < height &&
                        img_col >= 0 && img_col < width)
                    {
                        sum += input[img_row * width + img_col] *
                               kernel[k_row * kernel_size + k_col];
                    }
                }
            }

            output[row * width + col] = sum;
        }
    }
}

// GPU convolution wrapper
float *convolution_gpu(const float *input, const float *kernel, int width,
                       int height, int kernel_size, int method)
{
    float *d_input, *d_kernel, *d_output, *d_temp;
    float *h_output;

    size_t image_size = width * height * sizeof(float);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_input, image_size));
    CUDA_CHECK(cudaMalloc(&d_output, image_size));
    h_output = new float[width * height];

    // Copy input data
    CUDA_CHECK(cudaMemcpy(d_input, input, image_size, cudaMemcpyHostToDevice));

    // Setup grid and block dimensions
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);

    switch (method)
    {
    case 0: // Naive
        CUDA_CHECK(cudaMalloc(&d_kernel, kernel_bytes));
        CUDA_CHECK(cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice));
        convolution_naive<<<grid_size, block_size>>>(d_input, d_kernel, d_output,
                                                     width, height, kernel_size);
        CUDA_CHECK(cudaFree(d_kernel));
        break;

    case 1: // Shared memory
    {
        CUDA_CHECK(cudaMalloc(&d_kernel, kernel_bytes));
        CUDA_CHECK(cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice));
        int shared_size = (block_size.x + 2 * (kernel_size / 2)) *
                          (block_size.y + 2 * (kernel_size / 2)) * sizeof(float);
        convolution_shared_memory<<<grid_size, block_size, shared_size>>>(d_input, d_kernel, d_output, width, height, kernel_size);
        CUDA_CHECK(cudaFree(d_kernel));
    }
    break;

    case 2: // Constant memory
        CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, kernel, kernel_bytes));
        convolution_constant_memory<<<grid_size, block_size>>>(d_input, d_output, width, height, kernel_size);
        break;

    case 3: // Texture memory
    {
        // Create texture object
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = d_input;
        resDesc.res.pitch2D.width = width;
        resDesc.res.pitch2D.height = height;
        resDesc.res.pitch2D.pitchInBytes = width * sizeof(float);
        resDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.pitch2D.desc.x = 32;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;

        cudaTextureObject_t texObj;
        CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

        CUDA_CHECK(cudaMalloc(&d_kernel, kernel_bytes));
        CUDA_CHECK(cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice));

        convolution_texture<<<grid_size, block_size>>>(texObj, d_kernel, d_output, width, height, kernel_size);

        CUDA_CHECK(cudaDestroyTextureObject(texObj));
        CUDA_CHECK(cudaFree(d_kernel));
    }
    break;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return h_output;
}

// Generate Gaussian kernel
void generate_gaussian_kernel(float *kernel, int size, float sigma)
{
    int center = size / 2;
    float sum = 0.0f;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            float x = i - center;
            float y = j - center;
            kernel[i * size + j] = expf(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[i * size + j];
        }
    }

    // Normalize kernel
    for (int i = 0; i < size * size; i++)
    {
        kernel[i] /= sum;
    }
}

// Generate test image with patterns
void generate_test_image(float *image, int width, int height)
{
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            // Create a pattern with edges and gradients
            float x = (float)col / width;
            float y = (float)row / height;

            // Checkerboard pattern with sine waves
            image[row * width + col] =
                0.5f + 0.3f * sinf(20.0f * x) * sinf(20.0f * y) +
                0.2f * ((row / 50 + col / 50) % 2 == 0 ? 1.0f : -1.0f);
        }
    }
}

// Verify convolution results
bool verify_convolution(const float *cpu_result, const float *gpu_result,
                        int width, int height, float tolerance)
{
    for (int i = 0; i < width * height; i++)
    {
        float diff = fabsf(cpu_result[i] - gpu_result[i]);
        if (diff > tolerance)
        {
            return false;
        }
    }
    return true;
}
