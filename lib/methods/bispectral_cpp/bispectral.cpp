// For PI constant
#define _USE_MATH_DEFINES
// Header for this file
#include "bispectral.h"
// OpenMP
#include <omp.h>
// FFTW
#include <fftw3.h>
// Custom
#include "../../utils/utils_cpp/utils.h"


void bispectral_method(std::complex<double> *image, size_t h, size_t w, std::complex<double> const *otf, double alpha)
{
    // FFT
    fftw_plan p_forward = fftw_plan_dft_2d(h, w, reinterpret_cast<fftw_complex*>(image), reinterpret_cast<fftw_complex*>(image),
                                           FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p_forward);
    fftw_destroy_plan(p_forward);

    // Image size
    size_t size = h * w;

    // Init bispectrum and restored image phases
    double *bispectrum_phase = (double*)malloc(sizeof(double) * 9 * size);
    double *phi = (double*)malloc(sizeof(double) * size);

    // Half indexes
    size_t half_h = h / 2;
    size_t half_w = w / 2;

    // Proper bispectrum indexing
    size_t size_b = 3 * size;
    #define bisp_phi(u1, v1, i, j) bispectrum_phase[u1 * size_b + v1 * size + i * w + j]

    // Loop over all coordinates in bispectrum phase
    #pragma omp parallel for collapse(4)
    for (int u1 = 0; u1 < 3; u1++)
    {
        for (int v1 = 0; v1 < 3; v1++)
        {    
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    #define index_2_half_image(index, half_size) index != 2 ? index : half_size
                    #define roll_index(index, size, roll) (index + roll)%size

                    // Bispectrum phase formula
                    int u2 = index_2_half_image(u1, half_h);
                    int v2 = index_2_half_image(v1, half_w);
                    bisp_phi(u1, v1, i, j) = arg(image[u2 * w + v2] * image[i * w + j] *
                                                 conj(image[roll_index(i, h, u2) * w + roll_index(j, w, v2)]));

                    #undef index_2_half_image
                    #undef roll_index
                }
            }
        }
    }

    #define otf_mask(otf) (otf.real() < 0) * M_PI

    // phi(0, 0)
    phi[0] = bisp_phi(0, 0, 0, 0) - otf_mask(otf[0]);

    // phi(0, N/2)
    phi[half_w] = 0.5 * (bisp_phi(0, 2, half_h, half_w) + bisp_phi(0, 2, half_h, 0)) - 2 * otf_mask(otf[half_w]);

    // phi(0, 1)
    double sum = 0;
    #pragma omp parallel for reduction(+:sum)
    // Indexes are messed up for the reduction to work
    // for (int j = 1; j < half_w; j++)
    for (int j = size + 1; j < size + half_w; j++)
    {
        // sum += bisp_phi(0, 1, 0, j)
        sum += bispectrum_phase[j];
    }
    phi[1] = (2 / (double)w) * (phi[half_w] + otf_mask(otf[half_w]) + sum) - otf_mask(otf[1]);

    // [phi(0, 2), phi(0, N/2 - 1)]
    for (int l = 1; l < half_w-1; l++)
    {
        phi[l + 1] = phi[1] + phi[l] + otf_mask(otf[1]) + otf_mask(otf[l]) - otf_mask(otf[l + 1]) - bisp_phi(0, 1, 0, l);
    }

    // [phi(0, N/2 + 1), phi(0, N - 1)]
    #pragma omp parallel for
    for (int l = half_w+1; l < w; l++)
    {
        phi[l] = -phi[w - l];
    }

    // phi(N/2, 0)
    phi[half_h * w] = 0.5 * (bisp_phi(2, 0, half_h, half_w) + bisp_phi(2, 0, 0, half_w)) - 2 * otf_mask(otf[half_h * w]);

    // phi(1, 0)
    sum = 0;
    #pragma omp parallel for reduction(+:sum)
    // Indexes are messed up for the reduction to work
    //for (int i = 1; i < half_h; i++)
    for (int i = size_b + 1*w; i < size_b + half_h*w; i+=w)
    {
        //sum += bisp_phi(1, 0, i, 0);
        sum += bispectrum_phase[i];
    }
    phi[1 * w] = (2 / (double)h) * (phi[half_h * w] + otf_mask(otf[half_h * w]) + sum) - otf_mask(otf[1 * w]);

    // [phi(2, 0), phi(N/2 - 1, 0)]
    for (int k = 1; k < half_h-1; k++)
    {
        phi[(k + 1) * w] = phi[1 * w] + phi[k * w] + otf_mask(otf[1 * w]) + otf_mask(otf[k * w]) - otf_mask(otf[(k + 1) * w]) -
                           bisp_phi(1, 0, k, 0);
    }

    // [phi(N/2 + 1, 0), phi(N - 1, 0)]
    #pragma omp parallel for
    for (int k = half_h+1; k < h; k++)
    {
        phi[k * w] = -phi[(h - k)  * w];
    }

    // phi(N/2, N/2)
    phi[half_h * w + half_w] = 0.5 * (bisp_phi(2, 2, half_h, half_w) + bisp_phi(0, 0, 0, 0)) + otf_mask(otf[0]) -
                               otf_mask(otf[half_h * w + half_w]);
    
    // All of the rest phases in upper half
    for (int m = 1; m < half_h; m++)
    {
        for (int n = 0; n < w-1; n++)
        {
            phi[m * w + n + 1] = phi[1] + phi[m * w + n] + otf_mask(otf[1]) + otf_mask(otf[m * w + n]) -
                                 otf_mask(otf[m * w + n + 1]) - bisp_phi(0, 1, m, n);
        }
    }

    // Mirror phases to the bottom half (on both axes)
    #pragma omp parallel for collapse(2)
    for (int i = half_h; i < h; i++)
    {
        for (int j = 1; j < w; j++)
        {
            phi[i * w + j] = -phi[(h - i) * w + (w - j)];
        }
    }

    // Undef macros and free memory
    #undef otf_mask
    #undef bisp_phi
    free(bispectrum_phase);

    // Restore image FFT module
    #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        // Image FFT squared complex module
        std::complex<double> image_fft_squared_module = get_squared_complex_module(image[i]);
        // OTF squared complex module
        std::complex<double> otf_squared_module = get_squared_complex_module(otf[i]);
        
        // Formula
        image[i] = sqrt((image_fft_squared_module * otf_squared_module) / (otf_squared_module * otf_squared_module + alpha));
    }

    // Restore image FFT phase
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            image[i * w + j] = image[i * w + j] * exp(std::complex<double>(0, phi[i * w + j]));
        }
    }

    // Free memory
    free(phi);

    // IFFT
    fftw_plan p_backward = fftw_plan_dft_2d(h, w, reinterpret_cast<fftw_complex*>(image), reinterpret_cast<fftw_complex*>(image),
                                            FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p_backward);
    fftw_destroy_plan(p_backward);
}