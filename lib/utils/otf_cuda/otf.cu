// Header for this file
#include "otf.h"
// Math
#include <math_constants.h>
// Complex numbers for device
#include <cuComplex.h>
// CUDA FFT
#include <cufft.h>
// Custom
#include "../utils_cuda/utils.cu"


// --KERNEL--: computes pupil multiplied by complex exponent
__global__ void kernel_psf_prefft(cuDoubleComplex *psf_d, size_t h, size_t w, double a, double R2)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute index of the pixel
        size_t pixel_index = i*w + j;

        // "Radius" of the point (x, y) mapped to [-pi, pi]
        double x = ((double)j/(w/2) - 1) * CUDART_PI;
        double y = (1 - (double)i/(h/2)) * CUDART_PI;
        double r = x*x + y*y;

        // Pupil
        if (r > R2)
        {
            psf_d[pixel_index] = make_cuDoubleComplex(0, 0);
        }
        else
        {
            // Values inside pupil (complex exponent | formula is simplified due to the complex number having only imaginary part)
            sincos(a*r, &psf_d[pixel_index].y, &psf_d[pixel_index].x);
            // Full implementation:
            //psf_d[pixel_index] = make_cuDoubleComplex(0, a*(wlength - wlength0)*r);
            //float t = exp(psf_d[pixel_index].x);
            //sincos(psf_d[pixel_index].y, &psf_d[pixel_index].y, &psf_d[pixel_index].x);
            //psf_d[pixel_index].x *= t;
            //psf_d[pixel_index].y *= t;
        }
    }
}


// --KERNEL--: PSF squared complex module (with division by matrix size)
__global__ void kernel_psf_sqcm(cuDoubleComplex *psf_d, size_t h, size_t w, double psf_size)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute index of the pixel
        size_t pixel_index = i*w + j;

        // Divide by matrix size
        psf_d[pixel_index].x /= psf_size;
        psf_d[pixel_index].y /= psf_size;
        // Squared complex module
        psf_d[pixel_index] = cuCmul(psf_d[pixel_index], cuConj(psf_d[pixel_index]));
    }
}


/** PSF creation on device.
 * 
 * @param psf_d PSF on device
 * @param h PSF height
 * @param w PSF width
 * @param a defocus param
 * @param R pupil radius
 * @param numBlocks thread blocks
 * @param threadsPerBlock threads in a block
 * @param fftPlan FFT plan
*/
void create_psf_(cuDoubleComplex *psf_d, size_t h, size_t w, double a, double R,
                 const dim3 *numBlocks, const dim3 *threadsPerBlock, const cufftHandle *fftPlan)
{
    // Shift R to fit [-pi, pi] and save its square
    double R2 = (R * CUDART_PI) * (R * CUDART_PI);

    // Fill initial psf values
    kernel_psf_prefft<<<*numBlocks, *threadsPerBlock>>>(psf_d, h, w, a, R2);
    
    // FFT
    cufftExecZ2Z(*fftPlan, psf_d, psf_d, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    // Squared complex module
    kernel_psf_sqcm<<<*numBlocks, *threadsPerBlock>>>(psf_d, h, w, h*w);
}


/** PSF creation.
 * 
 * @param psf allocaded PSF memory
 * @param h PSF height
 * @param w PSF width
 * @param a defocus param
 * @param R pupil radius
*/
void create_psf(std::complex<double> *psf, size_t h, size_t w, double a, double R)
{
    // PSF memory size
    size_t psf_memsize = sizeof(std::complex<double>)*h*w;
    // Threads and thread blocks settings
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(h/threadsPerBlock.x + (h%threadsPerBlock.x != 0), w/threadsPerBlock.y + (w%threadsPerBlock.y != 0));
    // cuFFT plan
    cufftHandle fftPlan;
    cufftPlan2d(&fftPlan, h, w, CUFFT_Z2Z);

    // Allocate memory for PSF on device
    cuDoubleComplex *psf_d;
    cudaMalloc(&psf_d, psf_memsize);

    // Create PSF on device
    create_psf_(psf_d, h, w, a, R, &numBlocks, &threadsPerBlock, &fftPlan);

    // Retrieve result
    cudaMemcpy(psf, psf_d, psf_memsize, cudaMemcpyDeviceToHost);

    // Destroy FFT plan
    cufftDestroy(fftPlan);
    // Free allocated memory
    cudaFree(psf_d);
}


/** PSF to OTF conversion on device.
 * 
 * @param psf_d PSF on device
 * @param h PSF height
 * @param w PSF width
 * @param numBlocks thread blocks
 * @param threadsPerBlock threads in a block
 * @param fftPlan FFT plan
*/
void psf2otf(cuDoubleComplex *psf_d, size_t h, size_t w, const dim3 *numBlocks, const dim3 *threadsPerBlock, const cufftHandle *fftPlan)
{
    // FFT
    cufftExecZ2Z(*fftPlan, psf_d, psf_d, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    // According to formula
    cufftDoubleComplex max_val;
    cudaMemcpy(&max_val, psf_d, sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost); // copy otf[0,0]
    // Divide by maximum value
    kernel_mat_div_by_val<<<*numBlocks, *threadsPerBlock>>>(psf_d, h, w, max_val);
}


/** OTF creation.
 * 
 * @param otf allocaded OTF memory
 * @param h OTF height
 * @param w OTF width
 * @param a defocus param
 * @param R pupil radius
*/
void create_otf(std::complex<double> *otf, size_t h, size_t w, double a, double R)
{
    // OTF memory size
    size_t otf_memsize = sizeof(std::complex<double>)*h*w;
    // Threads and thread blocks settings
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(h/threadsPerBlock.x + (h%threadsPerBlock.x != 0), w/threadsPerBlock.y + (w%threadsPerBlock.y != 0));
    // cuFFT plan
    cufftHandle fftPlan;
    cufftPlan2d(&fftPlan, h, w, CUFFT_Z2Z);

    // Allocate memory for OTF on device
    cuDoubleComplex *otf_d;
    cudaMalloc(&otf_d, otf_memsize);

    // Create PSF on device
    create_psf_(otf_d, h, w, a, R, &numBlocks, &threadsPerBlock, &fftPlan);
    // Convert PSF to OTF
    psf2otf(otf_d, h, w, &numBlocks, &threadsPerBlock, &fftPlan);

    // Retrieve result
    cudaMemcpy(otf, otf_d, otf_memsize, cudaMemcpyDeviceToHost);

    // Destroy FFT plan
    cufftDestroy(fftPlan);
    // Free allocated memory
    cudaFree(otf_d);
}


// --KERNEL--: "Integrate" PSF stack, multiplied by apropriate detector functions
__global__ void kernel_integrate_psf_stack(cuDoubleComplex *psf_d, size_t h, size_t w, size_t psf_size, const cuDoubleComplex *psf_stack_d,
                                           const double *detector_funcs_d, size_t wlgrl, int psf_i, int psf_j, double wlgstep)
{
    // i and j are obtained via blocks and threads
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Verify that we are not exceeding array bounds
    if (i < h && j < w)
    {
        // Precompute index of the pixel
        size_t pixel_index = i*w + j;

        // Init with zero
        psf_d[pixel_index] = make_cuDoubleComplex(0,0);

        // "Integrate" using trapezoid formula for uniform grif
        for (size_t wl = 0; wl < wlgrl; wl++) // for every wave length
        {
            // Inside the wave length grid
            if ((wl != 0) && (wl != wlgrl-1))
            {
                // Step multiplied by function value
                psf_d[pixel_index] = cuCadd(psf_d[pixel_index], cuCmul(psf_stack_d[wl*psf_size + pixel_index],
                            make_cuDoubleComplex(wlgstep*detector_funcs_d[psf_i*wlgrl + wl]*detector_funcs_d[psf_j*wlgrl + wl], 0)));
            }
            // On the bounds of the wave length grid
            else 
            {
                // Step, divided by 2, multiplied by function value
                psf_d[pixel_index] = cuCadd(psf_d[pixel_index], cuCmul(psf_stack_d[wl*psf_size + pixel_index],
                            make_cuDoubleComplex((wlgstep/2)*detector_funcs_d[psf_i*wlgrl + wl]*detector_funcs_d[psf_j*wlgrl + wl], 0)));
            }
        }
    }
}


/** OTF batch creation.
 * 
 * @param otf_bacth Allocaded OTF batch memory
 * @param h OTF height
 * @param w OTF width
 * @param rgb_ratios RGB channel ratios that will define the image color
 * @param detector_funcs Wave length grid and detector functions of RGB channels
 * @param wlgrl Wave length grid length
 * @param a Defocus param
 * @param wlength0 Focus wave length
 * @param R Pupil radius
*/
void create_otf_batch(std::complex<double> *otf_bacth, size_t h, size_t w, const double* rgb_ratios, const double *detector_funcs,
                      size_t wlgrl, double a, double wlength0, double R)
{
    // OTF sizes
    size_t otf_size = h*w;
    size_t otf_memsize = sizeof(std::complex<double>)*otf_size;
    // Threads and thread blocks settings
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(h/threadsPerBlock.x + (h%threadsPerBlock.x != 0), w/threadsPerBlock.y + (w%threadsPerBlock.y != 0));
    // cuFFT plan
    cufftHandle fftPlan;
    cufftPlan2d(&fftPlan, h, w, CUFFT_Z2Z);

    // Allocate memory for PSF stack on device
    cuDoubleComplex *psf_stack_d;
    cudaMalloc(&psf_stack_d, otf_memsize*wlgrl);
    // Allocate memory for detector functions on device
    double *detector_funcs_d;
    cudaMalloc(&detector_funcs_d, sizeof(double)*3*wlgrl);
    // Allocate memory for OTF on device
    cuDoubleComplex *otf_d;
    cudaMalloc(&otf_d, otf_memsize);

    // Send functions (without wave length grid) to device
    cudaMemcpy(detector_funcs_d, detector_funcs+wlgrl, sizeof(double)*3*wlgrl, cudaMemcpyHostToDevice);

    // Fill PSF stack
    for (size_t wl = 0; wl < wlgrl; wl++)
    {
        // Create PSF on device (with respect to wave length)
        create_psf_(psf_stack_d + wl*otf_size, h, w, a * (detector_funcs[wl] - wlength0), R, &numBlocks, &threadsPerBlock, &fftPlan);
    }

    // "Integrate" PSF stack, multiplied by apropriate detector functions, and convert result to OTF
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            // Create PSF in position (i,j) of the batch
            kernel_integrate_psf_stack<<<numBlocks, threadsPerBlock>>>(otf_d, h, w, otf_size, psf_stack_d, detector_funcs_d,
                                                                       wlgrl, i, j, detector_funcs[1] - detector_funcs[0]);

            // Convert PSF to OTF
            psf2otf(otf_d, h, w, &numBlocks, &threadsPerBlock, &fftPlan);

            // Multiply by RGB ratios to encrypt color information in OTF batch
            kernel_mat_mul_by_val<<<numBlocks, threadsPerBlock>>>(otf_d, h, w, make_cuDoubleComplex(rgb_ratios[i] * rgb_ratios[j], 0));

            // Retrieve result
            cudaMemcpy(otf_bacth + (i*3 + j)*otf_size, otf_d, otf_memsize, cudaMemcpyDeviceToHost);
        }
    }

    // Destroy FFT plan
    cufftDestroy(fftPlan);
    // Free allocated memory
    cudaFree(psf_stack_d);
    cudaFree(detector_funcs_d);
    cudaFree(otf_d);
}