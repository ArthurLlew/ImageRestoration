// Header for this file
#include "otf_restoration.h"
// Math
#include "math_constants.h"
// Complex numbers for device
#include <cuComplex.h>
// CUDA FFT
#include <cufft.h>


// --KERNEL--: complex module
__global__ void kernel_complex_abs(cuDoubleComplex *image_d, size_t h, size_t w)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute index of the pixel
        size_t pixel_index = i * w + j;

        // Complex module
        image_d[pixel_index] = make_cuDoubleComplex(cuCabs(image_d[pixel_index]), 0);
    }
}


// --KERNEL--: average angled spectrum
__global__ void kernel_average_angled_spectrum(cuDoubleComplex *image_d, size_t h, size_t w, double *av_ang_spec, size_t image_half_size,
                                               double k, double b)
{
    // Averaged angled spectrum index
    int s = blockIdx.x * 16 + threadIdx.x;
    // Verify that we are not exceeding array bounds
    if (s < image_half_size)
    {
        // Init
        int divider = 0;
        av_ang_spec[s] = 0;

        // Sum of values in a ring
        for (size_t i = 0; i < h; i++)
        {
            for (size_t j = 0; j < w; j++)
            {
                // Point coordinates in pixels (includes shift)
                double i_shifted = (i + (h/2)) % h;
                double j_shifted = (j + (w/2)) % w;
                double x = (j_shifted - image_half_size);
                double y = (image_half_size - i_shifted);
                // Point radius in pixels
                double r = x*x + y*y;

                // Verify that we are in a ring
                if ((s*s <= r) && (r < (s+1)*(s+1)))
                {
                    divider++;
                    av_ang_spec[s] += image_d[i*w + j].x;
                }
            }
        }

        // Averaged angled spectrum
        av_ang_spec[s] /= divider;

        // Scaled arctan from averaged angled spectrum multiplied by linear function
        av_ang_spec[s] = atan(av_ang_spec[s] * (k * (s * CUDART_PI / image_half_size) + b));
    }
}


void get_average_angled_spectrum(std::complex<double> *image, size_t h, size_t w, double *av_ang_spec, size_t image_half_size,
                                 double k, double b)
{
    // Sizes
    size_t image_size = h*w;
    size_t image_memsize = sizeof(std::complex<double>)*image_size;
    size_t av_ang_spec_memsize = image_half_size*sizeof(double);
    // Threads and blocks settings
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(h/threadsPerBlock.x + (h%threadsPerBlock.x != 0), w/threadsPerBlock.y + (w%threadsPerBlock.y != 0));
    // cuFFT plan
    cufftHandle fftPlan;
    cufftPlan2d(&fftPlan, h, w, CUFFT_Z2Z);

    // Allocate memory for the image on device
    cuDoubleComplex *image_d;
    cudaMalloc(&image_d, image_memsize);
    // Allocate memory for the average angled spectrum on device
    double *av_ang_spec_d;
    cudaMalloc(&av_ang_spec_d, av_ang_spec_memsize);

    // Send image to the device
    cudaMemcpy(image_d, image, image_memsize, cudaMemcpyHostToDevice);

    // FFT
    cufftExecZ2Z(fftPlan, image_d, image_d, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    // Complex module
    kernel_complex_abs<<<numBlocks, threadsPerBlock>>>(image_d, h, w);

    // Compute average angled spectrum
    unsigned int blocks = image_half_size/threadsPerBlock.x + (image_half_size%threadsPerBlock.x != 0);
    kernel_average_angled_spectrum<<<blocks, threadsPerBlock.x>>>(image_d, h, w, av_ang_spec_d, image_half_size, k, b);

    // Retrieve result
    cudaMemcpy(av_ang_spec, av_ang_spec_d, av_ang_spec_memsize, cudaMemcpyDeviceToHost);

    // Destroy FFT plan
    cufftDestroy(fftPlan);
    // Free remaining allocated memory
    cudaFree(image_d);
    cudaFree(av_ang_spec_d);
}