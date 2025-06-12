// Complex numbers for device
#include <cuComplex.h>


// --KERNEL--: matrix subtraction by value (double)
__global__ void kernel_mat_sub_by_val(double *matrix, size_t h, size_t w, double value)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute index of the pixel
        size_t pixel_index = i*w + j;

        // Subtract by provided value
        matrix[pixel_index] -= value;
    }
}


// --KERNEL--: matrix subtraction by value (complex)
__global__ void kernel_mat_sub_by_val(cuDoubleComplex *matrix, size_t h, size_t w, cuDoubleComplex value)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute index of the pixel
        size_t pixel_index = i*w + j;

        // Subtract by provided value
        matrix[pixel_index] = cuCsub(matrix[pixel_index], value);
    }
}


// --KERNEL--: matrix multiplication by value (double)
__global__ void kernel_mat_mul_by_val(double *matrix, size_t h, size_t w, double value)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute index of the pixel
        size_t pixel_index = i*w + j;

        // Multiply by provided value
        matrix[pixel_index] *= value;
    }
}


// --KERNEL--: matrix multiplication by value (complex)
__global__ void kernel_mat_mul_by_val(cuDoubleComplex *matrix, size_t h, size_t w, cuDoubleComplex value)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute index of the pixel
        size_t pixel_index = i*w + j;

        // Multiply by provided value
        matrix[pixel_index] = cuCmul(matrix[pixel_index], value);
    }
}


// --KERNEL--: matrix division by value (double)
__global__ void kernel_mat_div_by_val(double *matrix, size_t h, size_t w, double value)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute index of the pixel
        size_t pixel_index = i*w + j;

        // Divide by provided value
        matrix[pixel_index] /= value;
    }
}


// --KERNEL--: matrix division by value (complex)
__global__ void kernel_mat_div_by_val(cuDoubleComplex *matrix, size_t h, size_t w, cuDoubleComplex value)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute index of the pixel
        size_t pixel_index = i*w + j;

        // Divide by provided value
        matrix[pixel_index] = cuCdiv(matrix[pixel_index], value);
    }
}


// --KERNEL--: clamps image
__global__ void kernel_clamp_image(double *image_d, double new_min, double new_max, size_t h, size_t w)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute indexes
        size_t pixel_index = i*w + j; // pixel index

        // Clamp pixel value
        if (image_d[pixel_index] < new_min)
        {
            image_d[pixel_index] = new_min;
        }
        else if (image_d[pixel_index] > new_max)
        {
            image_d[pixel_index] = new_max;
        }
    }
}