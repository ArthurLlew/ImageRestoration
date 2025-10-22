// For PI constant
#define _USE_MATH_DEFINES
// Header for this file
#include "tikhonov_rgb.h"
// Math
#include <math_constants.h>
// Complex numbers for device
#include <cuComplex.h>
// CUDA FFT
#include <cufft.h>
// CUDA CUB library
#include <cub/cub.cuh>
// Custom
#include "../../utils/utils_cuda/utils.cu"
#include "../distortion_cuda/distortion.cu"


// --KERNEL_FUNC--: computes 3x3 matrix and vector multiplication
void __device__ device_matr3x3_mul_vec3(const cuDoubleComplex *matr, cuDoubleComplex *vec)
{
    // 3x3 matrix multiplication by vector
    cuDoubleComplex matrmul[3] = {make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,0)};
    for (int i = 0; i < 3; i++)
    {
        // Calculate element (i,0)
        for (int ij = 0; ij < 3; ij++)
        {
            matrmul[i] = cuCadd(matrmul[i], cuCmul(matr[i*3 + ij], vec[ij]));
        }
    }
    // Save result
    memcpy(vec, matrmul, sizeof(cuDoubleComplex)*3);
}


// --KERNEL--: computes the "inverse matrix" and the 2nd term in method formula
__global__ void kernel_prepair_method(cuDoubleComplex *image_rest_d, cuDoubleComplex *otf_batch_d, cuDoubleComplex *invmatr_d,
                                      cuDoubleComplex *term2_d, const double *mu_d, size_t h, size_t w)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute indexes
        size_t pixel_index = i*w + j; // pixel index
        size_t rgb_pixel_index = pixel_index*3; // each RGB pixel holds 3 values
        size_t batch_index = pixel_index*9; // in batch each "pixel" is a 3x3 matrix

        // Init restored image as zero matrix
        for (int channel = 0; channel < 3; channel++)
        {
            image_rest_d[rgb_pixel_index + channel] = make_cuDoubleComplex(0,0);
        }

        // Conjugate the OTF batch 3x3 matrix
        for (int i_batch = 0; i_batch < 3; i_batch++)
        {
            for (int j_batch = 0; j_batch < 3; j_batch++)
            {
                otf_batch_d[batch_index + i_batch*3 + j_batch] = cuConj(otf_batch_d[batch_index + i_batch*3 + j_batch]);
            }
        }

        // Matrix multiplication of OTF batch 3x3 matrix and conjugated OTF batch 3x3 matrix (is also multiplied by 'mu')
        cuDoubleComplex matrmul[9] = {make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,0),
                                      make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,0),
                                      make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,0)};
        for (int i_batch = 0; i_batch < 3; i_batch++)
        {
            for (int j_batch = 0; j_batch < 3; j_batch++)
            {
                // Calculate element (i,j)
                for (int ij = 0; ij < 3; ij++)
                {
                    matrmul[i_batch*3 + j_batch] = cuCadd(matrmul[i_batch*3 + j_batch],
                                             cuCmul(otf_batch_d[batch_index + i_batch*3 + ij], invmatr_d[batch_index + ij*3 + j_batch]));
                }

                 //Is this legal??? (numpy does this for some reason)
                if (i_batch == j_batch)
                {
                    matrmul[i_batch*3 + j_batch].y = 0;
                }

                // Multiply by mu[u,v]
                matrmul[i_batch*3 + j_batch].x *= mu_d[pixel_index];
                matrmul[i_batch*3 + j_batch].y *= mu_d[pixel_index];
            }
        }
        // Save result
        memcpy(invmatr_d + batch_index, matrmul, sizeof(cuDoubleComplex)*9);

        // Add EYE 3x3 matix
        invmatr_d[batch_index].x += 1;
        invmatr_d[batch_index+4].x += 1;
        invmatr_d[batch_index+8].x += 1;
        
        // Matrix determinant
        cuDoubleComplex matrdet = cuCmul(cuCmul(invmatr_d[batch_index], invmatr_d[batch_index+4]), invmatr_d[batch_index+8]);
        matrdet = cuCadd(matrdet, cuCmul(cuCmul(invmatr_d[batch_index+2], invmatr_d[batch_index+3]), invmatr_d[batch_index+7]));
        matrdet = cuCadd(matrdet, cuCmul(cuCmul(invmatr_d[batch_index+1], invmatr_d[batch_index+5]), invmatr_d[batch_index+6]));
        matrdet = cuCsub(matrdet, cuCmul(cuCmul(invmatr_d[batch_index+2], invmatr_d[batch_index+4]), invmatr_d[batch_index+6]));
        matrdet = cuCsub(matrdet, cuCmul(cuCmul(invmatr_d[batch_index], invmatr_d[batch_index+5]), invmatr_d[batch_index+7]));
        matrdet = cuCsub(matrdet, cuCmul(cuCmul(invmatr_d[batch_index+1], invmatr_d[batch_index+3]), invmatr_d[batch_index+8]));
        // Inverse matrix
        cuDoubleComplex invmatr[9];
        invmatr[0] = cuCsub(cuCmul(invmatr_d[batch_index+4], invmatr_d[batch_index+8]),
                            cuCmul(invmatr_d[batch_index+5], invmatr_d[batch_index+7])); // (0,0)
        invmatr[1] = cuCsub(cuCmul(invmatr_d[batch_index+3], invmatr_d[batch_index+8]),
                            cuCmul(invmatr_d[batch_index+5], invmatr_d[batch_index+6])); // (0,1)
        invmatr[2] = cuCsub(cuCmul(invmatr_d[batch_index+3], invmatr_d[batch_index+7]),
                            cuCmul(invmatr_d[batch_index+4], invmatr_d[batch_index+6])); // (0,2)
        invmatr[3] = cuCsub(cuCmul(invmatr_d[batch_index+1], invmatr_d[batch_index+8]),
                            cuCmul(invmatr_d[batch_index+2], invmatr_d[batch_index+7])); // (1,0)
        invmatr[4] = cuCsub(cuCmul(invmatr_d[batch_index], invmatr_d[batch_index+8]),
                            cuCmul(invmatr_d[batch_index+2], invmatr_d[batch_index+6])); // (1,1)
        invmatr[5] = cuCsub(cuCmul(invmatr_d[batch_index], invmatr_d[batch_index+7]),
                            cuCmul(invmatr_d[batch_index+1], invmatr_d[batch_index+6])); // (1,1)
        invmatr[6] = cuCsub(cuCmul(invmatr_d[batch_index+1], invmatr_d[batch_index+5]),
                            cuCmul(invmatr_d[batch_index+2], invmatr_d[batch_index+4])); // (2,0)
        invmatr[7] = cuCsub(cuCmul(invmatr_d[batch_index], invmatr_d[batch_index+5]),
                            cuCmul(invmatr_d[batch_index+2], invmatr_d[batch_index+3])); // (2,1)
        invmatr[8] = cuCsub(cuCmul(invmatr_d[batch_index], invmatr_d[batch_index+4]),
                            cuCmul(invmatr_d[batch_index+1], invmatr_d[batch_index+3])); // (2,2)
        for (int i_batch = 0; i_batch < 3; i_batch++)
        {
            for (int j_batch = 0; j_batch < 3; j_batch++)
            {
                invmatr[i_batch*3 + j_batch].x *= pow(-1, double(i_batch + j_batch));
                invmatr[i_batch*3 + j_batch].y *= pow(-1, double(i_batch + j_batch));
                invmatr[i_batch*3 + j_batch] = cuCdiv(invmatr[i_batch*3 + j_batch], matrdet);
            }
        }
        // Save result
        memcpy(invmatr_d + batch_index, invmatr, sizeof(cuDoubleComplex)*9);

        // Multiply conjugated OTF batch 3x3 matrix by observed image FFT
        device_matr3x3_mul_vec3(otf_batch_d + batch_index, term2_d + rgb_pixel_index);
        // Multiply inverse matrix by the resulting vector
        device_matr3x3_mul_vec3(invmatr_d + batch_index, term2_d + rgb_pixel_index);
        // Multiply resulting vector by 'mu'
        for (int channel = 0; channel < 3; channel++)
        {
            term2_d[rgb_pixel_index + channel].x *= mu_d[pixel_index];
            term2_d[rgb_pixel_index + channel].y *= mu_d[pixel_index];
        }
    }
}


// --KERNEL--: method step
__global__ void kernel_method_step(cuDoubleComplex *image_rest_d, const cuDoubleComplex *invmatr_d, const cuDoubleComplex *term2_d, 
                                   size_t h, size_t w)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute indexes
        size_t pixel_index = i*w + j; // pixel index
        size_t rgb_pixel_index = pixel_index*3; // each RGB pixel holds 3 values
        size_t batch_index = pixel_index*9; // in batch each "pixel" is a 3x3 matrix

        // Multiply inverse matrix by restored image fft from previous step
        device_matr3x3_mul_vec3(invmatr_d + batch_index, image_rest_d + rgb_pixel_index);
        // Add second term from formula
        for (int channel = 0; channel < 3; channel++)
        {
            image_rest_d[rgb_pixel_index + channel] = cuCadd(image_rest_d[rgb_pixel_index + channel], term2_d[rgb_pixel_index + channel]);
        }
    }
}


// --KERNEL--: rearranges image in different storage
__global__ void kernel_rearrange_image(const cuDoubleComplex *temp_image_1_d, double *temp_image_2_d, size_t h, size_t w,
                                       size_t image_size)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute indexes
        size_t pixel_index = i*w + j; // pixel index
        size_t rgb_pixel_index = pixel_index*3; // each RGB pixel holds 3 values

        // Rearrange RGB pixels
        for (int channel = 0; channel < 3; channel++)
        {
            temp_image_2_d[channel * image_size + pixel_index] = temp_image_1_d[rgb_pixel_index + channel].x;
        }
    }
}


// --KERNEL--: swaps images
__global__ void kernel_swap_images(cuDoubleComplex *temp_image_1_d, double *temp_image_2_d, size_t h, size_t w,
                                   size_t image_size)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute indexes
        size_t pixel_index = i*w + j; // pixel index
        size_t rgb_pixel_index = pixel_index*3; // each RGB pixel holds 3 values

        // Swap RGB pixels
        for (int channel = 0; channel < 3; channel++)
        {
            double value = temp_image_2_d[channel * image_size + pixel_index];
            temp_image_2_d[channel * image_size + pixel_index] = temp_image_1_d[rgb_pixel_index + channel].x;
            temp_image_1_d[rgb_pixel_index + channel] = make_cuDoubleComplex(value, 0);
        }
    }
}


// --KERNEL--: computes images discrepancy square
__global__ void kernel_discrepancy_square(cuDoubleComplex *temp_image_1_d, double *temp_image_2_d, size_t h, size_t w,
                                          size_t image_size)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute indexes
        size_t pixel_index = i*w + j; // pixel index
        size_t rgb_pixel_index = pixel_index*3; // each RGB pixel holds 3 values

        // Swap RGB pixels
        for (int channel = 0; channel < 3; channel++)
        {
            // Discrepancy
            temp_image_2_d[channel * image_size + pixel_index] = temp_image_2_d[channel * image_size + pixel_index] -
                                                                 temp_image_1_d[rgb_pixel_index + channel].x;
            // Square
            temp_image_2_d[channel * image_size + pixel_index] = temp_image_2_d[channel * image_size + pixel_index] *
                                                                 temp_image_2_d[channel * image_size + pixel_index];
        }
    }
}


/** Normalizes RGB image.
 * 
 * @param image_d image on device
 * @param h image height
 * @param w image width
 * @param rgb_image_size image size
 * @param min_max_d min/max device memory
 * @param min_d min device memory
 * @param min_d_memsize min device memory size
 * @param max_d max device memory
 * @param max_d_memsize max device memory size
 * @param threadsPerBlock min/man device memory
 * @param numBlocks min/man device memory
*/
void normalize_image(double *image_d, size_t h, size_t w, size_t image_size, size_t rgb_image_size, double *min_max_d,
                     void *min_d, size_t min_d_memsize, void *max_d, size_t max_d_memsize,
                     dim3 threadsPerBlock, dim3 numBlocks)
{
    // Min/max storage
    double min_max;

    // Subtract global min
    cub::DeviceReduce::Min(min_d, min_d_memsize, image_d, min_max_d, rgb_image_size);
    cudaMemcpy(&min_max, min_max_d, sizeof(double), cudaMemcpyDeviceToHost);
    for (int ch = 0; ch < 3; ch++)
    {
        kernel_mat_sub_by_val<<<numBlocks, threadsPerBlock>>>(image_d + ch*image_size, h, w, min_max);
    }

    // Divide by global max
    cub::DeviceReduce::Max(max_d, max_d_memsize, image_d, min_max_d, rgb_image_size);
    cudaMemcpy(&min_max, min_max_d, sizeof(double), cudaMemcpyDeviceToHost);
    for (int ch = 0; ch < 3; ch++)
    {
        kernel_mat_div_by_val<<<numBlocks, threadsPerBlock>>>(image_d + ch*image_size, h, w, min_max);
    }
}


/**
 * Iterative tikhonov regularization method of RGB image restoration.
 * 
 * @param image image to restore
 * @param h image height
 * @param w image width
 * @param otf_batch OTF batch
 * @param mu regularization parameter
 * @param k0 number of method iterations
**/
void tikhonov_regularization_method_rgb(std::complex<double> *image, size_t h, size_t w,
                                        const std::complex<double> *otf_batch,
                                        const double *mu, int k0, double *discrepancy)
{
    // (0 iterations means "do nothing")
    if (k0 != 0)
    {
        // Sizes
        size_t image_size = h*w;
        size_t rgb_image_size = 3*image_size;
        size_t image_memsize = sizeof(std::complex<double>)*image_size;
        size_t image_memsize_ = sizeof(double)*image_size;
        size_t rgb_image_memsize = image_memsize*3;
        size_t rgb_image_memsize_ = image_memsize_*3;
        size_t otf_batch_memsize = image_memsize*9;
        // Threads and blocks settings
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks(h/threadsPerBlock.x + (h%threadsPerBlock.x != 0), w/threadsPerBlock.y + (w%threadsPerBlock.y != 0));
        // cuFFT plan
        cufftHandle fftPlan;
        int n[2] = {(int)h, (int)w};    
        cufftPlanMany(&fftPlan, 2, n, n, 3, 1, n, 3, 1, CUFFT_Z2Z, 3);

        // Allocate memory for the restored image
        cuDoubleComplex *image_rest_d;
        cudaMalloc(&image_rest_d, rgb_image_memsize);
        // Allocate memory for the OTF batch on device
        cuDoubleComplex *otf_batch_d;
        cudaMalloc(&otf_batch_d, otf_batch_memsize);
        // Allocate memory for the "inverse matrix" on device
        cuDoubleComplex *invmatr_d;
        cudaMalloc(&invmatr_d, otf_batch_memsize);
        // Allocate memory for the 2nd term in method formula
        cuDoubleComplex *term2_d;
        cudaMalloc(&term2_d, rgb_image_memsize);
        // Allocate memory for the mu parameter in method formula
        double *mu_d;
        cudaMalloc(&mu_d, image_memsize_);

        // Send image to the device
        cudaMemcpy(term2_d, image, rgb_image_memsize, cudaMemcpyHostToDevice);
        // Send two copies of OTF batch to device
        cudaMemcpy(otf_batch_d, otf_batch, otf_batch_memsize, cudaMemcpyHostToDevice);
        cudaMemcpy(invmatr_d, otf_batch, otf_batch_memsize, cudaMemcpyHostToDevice);
        // Send 'mu' to the device
        cudaMemcpy(mu_d, mu, image_memsize_, cudaMemcpyHostToDevice);

        // FFT of the observed RGB image
        cufftExecZ2Z(fftPlan, term2_d, term2_d, CUFFT_FORWARD);
        cudaDeviceSynchronize();
        // Prepair the "inverse matrix" and the 2nd term in method formula
        kernel_prepair_method<<<numBlocks, threadsPerBlock>>>(image_rest_d, otf_batch_d, invmatr_d, term2_d, mu_d, h, w);

        // Free 'mu' memory on device
        cudaFree(mu_d);

        // k0 > 0 means fixed number of iterations
        if (k0 > 0)
        {
            // Free OTF batch
            cudaFree(otf_batch_d);

            // Method iterations
            for (int i = 0; i < k0; i++)
            {
                // Perform method step
                kernel_method_step<<<numBlocks, threadsPerBlock>>>(image_rest_d, invmatr_d, term2_d, h, w);
            }
        }
        // k0 < 0 means max number of iterations
        else
        {
            k0 = -k0;

            // Allocate temporary images memory
            cuDoubleComplex *image_blurred_d;
            cudaMalloc(&image_blurred_d, rgb_image_memsize);
            cudaMemcpy(image_blurred_d, image, rgb_image_memsize, cudaMemcpyHostToDevice);
            cuDoubleComplex *temp_image_1_d;
            cudaMalloc(&temp_image_1_d, rgb_image_memsize);
            double *temp_image_2_d;
            cudaMalloc(&temp_image_2_d, rgb_image_memsize_);

            // Allocate memory for max/min value
            double *min_max_d;
            cudaMalloc(&min_max_d, sizeof(double));
            // Determine and allocate min temporary device storage
            void *min_d = nullptr;
            size_t min_d_memsize = 0;
            cub::DeviceReduce::Min(min_d, min_d_memsize, temp_image_2_d, min_max_d, rgb_image_size);
            cudaMalloc(&min_d, min_d_memsize);
            // Determine and allocate max temporary device storage
            void *max_d = nullptr;
            size_t max_d_memsize = 0;
            cub::DeviceReduce::Max(max_d, max_d_memsize, temp_image_2_d, min_max_d, rgb_image_size);
            cudaMalloc(&max_d, max_d_memsize);

            // Histogram threshold
            int threshold_px = int(0.005 * image_size);

            // Allocate memory for histogram
            size_t histogram_memsize = sizeof(int)*256;
            int* histogram = (int*)malloc(histogram_memsize);
            int* histogram_d;
            cudaMalloc(&histogram_d, histogram_memsize);
            // Determine and allocate histogram temporary device storage
            void* histogram_d_temp = nullptr;
            size_t histogram_d_temp_memsize = 0;
            cub::DeviceHistogram::HistogramEven(histogram_d_temp, histogram_d_temp_memsize, temp_image_2_d, histogram_d, 257,
                                                0.0, 1.0, image_size);
            cudaMalloc(&histogram_d_temp, histogram_d_temp_memsize);

            // Integration step
            double h_ij = (M_PI/h)*(M_PI/w);

            // Allocate memory for integral value
            double integral;
            double *integral_d;
            cudaMalloc(&integral_d, sizeof(double));
            // Determine and allocate integral value temporary device storage
            void *integral_d_temp = nullptr;
            size_t integral_d_temp_memsize = 0;
            cub::DeviceReduce::Sum(integral_d_temp, integral_d_temp_memsize, temp_image_2_d, integral_d, image_size);
            cudaMalloc(&integral_d_temp, integral_d_temp_memsize);

            // Method iterations
            int i = 0;
            while (i < k0)
            {
                // Perform method step
                kernel_method_step<<<numBlocks, threadsPerBlock>>>(image_rest_d, invmatr_d, term2_d, h, w);

                // Copy restored image
                cudaMemcpy(temp_image_1_d, image_rest_d, rgb_image_memsize, cudaMemcpyDeviceToDevice);
                // Perform IFFT
                cufftExecZ2Z(fftPlan, temp_image_1_d, temp_image_1_d, CUFFT_INVERSE);
                cudaDeviceSynchronize();
                // Rearrange image
                kernel_rearrange_image<<<numBlocks, threadsPerBlock>>>(temp_image_1_d, temp_image_2_d, h, w, image_size);
                // Normalize image
                normalize_image(temp_image_2_d, h, w, image_size, rgb_image_size, min_max_d, min_d, min_d_memsize, max_d,
                                max_d_memsize, threadsPerBlock, numBlocks);
                // Perform autocontrast
                for (int ch = 0; ch < 3; ch++)
                {
                    // Get image channel histogram
                    cub::DeviceHistogram::HistogramEven(histogram_d_temp, histogram_d_temp_memsize, temp_image_2_d + ch*image_size,
                                                        histogram_d, 257, 0.0, 1.0, image_size);
                    cudaMemcpy(histogram, histogram_d, histogram_memsize, cudaMemcpyDeviceToHost);
                    // New image channel minimum from histogram
                    int j = 0;
                    while (j < 256 && histogram[j] < threshold_px)
                    {
                        j++;
                    }
                    double new_min = j/256.0;
                    // New image channel maximum from histogram
                    j = 255;
                    while (j > 0 && histogram[j] < threshold_px)
                    {
                        j--;
                    }
                    double new_max = j/256.0;
                    // Clamp image channel
                    kernel_clamp_image<<<numBlocks, threadsPerBlock>>>(temp_image_2_d + ch*image_size, new_min, new_max, h, w);
                }
                // Normalize image
                normalize_image(temp_image_2_d, h, w, image_size, rgb_image_size, min_max_d, min_d, min_d_memsize, max_d,
                                max_d_memsize, threadsPerBlock, numBlocks);

                // Move image back
                kernel_swap_images<<<numBlocks, threadsPerBlock>>>(temp_image_1_d, temp_image_2_d, h, w, image_size);
                // Perform FFT
                cufftExecZ2Z(fftPlan, temp_image_1_d, temp_image_1_d, CUFFT_FORWARD);
                cudaDeviceSynchronize();

                // Blur image
                kernel_blur_image<<<numBlocks, threadsPerBlock>>>(temp_image_1_d, otf_batch_d, h, w);
                // Perform IFFT
                cufftExecZ2Z(fftPlan, temp_image_1_d, temp_image_1_d, CUFFT_INVERSE);
                cudaDeviceSynchronize();

                // Swap images
                kernel_swap_images<<<numBlocks, threadsPerBlock>>>(temp_image_1_d, temp_image_2_d, h, w, image_size);
                // Normalize image
                normalize_image(temp_image_2_d, h, w, image_size, rgb_image_size, min_max_d, min_d, min_d_memsize, max_d,
                                max_d_memsize, threadsPerBlock, numBlocks);

                // Compute discrepancy square
                kernel_discrepancy_square<<<numBlocks, threadsPerBlock>>>(image_blurred_d, temp_image_2_d, h, w, image_size);
                // Integrate each channel separately
                for (int ch = 0; ch < 3; ch++)
                {
                    // Integrate
                    kernel_mat_mul_by_val<<<numBlocks, threadsPerBlock>>>(temp_image_2_d + ch*image_size, h, w, h_ij);
                    cub::DeviceReduce::Sum(integral_d_temp, integral_d_temp_memsize, temp_image_2_d + ch*image_size,
                                           integral_d, image_size);
                    // Sqrt of integral value
                    cudaMemcpy(&integral, integral_d, sizeof(double), cudaMemcpyDeviceToHost);
                    discrepancy[i] += sqrt(integral);
                }

                // Update iteration number
                i++;
            }

            // Free temp data
            cudaFree(min_max_d);
            cudaFree(min_d);
            cudaFree(max_d);
            free(histogram);
            cudaFree(histogram_d);
            cudaFree(histogram_d_temp);
            cudaFree(integral_d);
            cudaFree(integral_d_temp);
            // Free temp images
            cudaFree(image_blurred_d);
            cudaFree(temp_image_1_d);
            cudaFree(temp_image_2_d);

            // Free OTF batch
            cudaFree(otf_batch_d);
        }

        // IFFT
        cufftExecZ2Z(fftPlan, image_rest_d, image_rest_d, CUFFT_INVERSE);
        cudaDeviceSynchronize();

        // Retrieve result
        cudaMemcpy(image, image_rest_d, rgb_image_memsize, cudaMemcpyDeviceToHost);

        // Destroy FFT plan
        cufftDestroy(fftPlan);
        // Free remaining memory
        cudaFree(image_rest_d);
        cudaFree(invmatr_d);
        cudaFree(term2_d);
    }
}