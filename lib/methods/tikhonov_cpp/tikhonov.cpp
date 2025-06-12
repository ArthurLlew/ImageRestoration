// For PI constant
#define _USE_MATH_DEFINES
// Header for this file
#include "tikhonov.h"
// OpenMP
#include <omp.h>
// FFTW
#include <fftw3.h>
// Custom
#include "../../utils/utils_cpp/utils.h"


void tikhonov_regularization_method(std::complex<double> *image, size_t h, size_t w, std::complex<double> const *otf,
                                    double alpha, double r)
{
    // FFT
    fftw_plan p_forward = fftw_plan_dft_2d(h, w, reinterpret_cast<fftw_complex*>(image), reinterpret_cast<fftw_complex*>(image),
                                           FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p_forward);
    fftw_destroy_plan(p_forward);

    // Tikhonov regularization
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            // Point coordinates (with shift)
            double i_shifted = (i + (h/2)) % h;
            double j_shifted = (j + (w/2)) % w;
            double x = (j_shifted/(w/2) - 1) * M_PI;
            double y = (1 - i_shifted/(h/2)) * M_PI;
            // Formula
            std::complex<double> upper_part = std::conj(otf[i * w + j]) * image[i * w + j];
            std::complex<double> down_part1 = get_squared_complex_module(otf[i * w + j]);
            std::complex<double> down_part2 = alpha * pow(x*x + y*y, r);
            image[i * w + j] = upper_part / (down_part1 + down_part2);
        }
    }

    // IFFT
    fftw_plan p_backward = fftw_plan_dft_2d(h, w, reinterpret_cast<fftw_complex*>(image), reinterpret_cast<fftw_complex*>(image),
                                            FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p_backward);
    fftw_destroy_plan(p_backward);
}