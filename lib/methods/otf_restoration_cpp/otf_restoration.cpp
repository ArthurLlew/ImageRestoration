// For PI constant
#define _USE_MATH_DEFINES
// Header for this file
#include "otf_restoration.h"
// OpenMP
#include <omp.h>
// FFTW
#include <fftw3.h>
// OTF
#ifdef DLL
  #undef DLL
  #include "../../utils/otf_cpp/otf.h"
  #define DLL
#else
  #include "../../utils/otf_cpp/otf.h"
#endif


void get_average_angled_spectrum(std::complex<double> *image, size_t h, size_t w, double *av_ang_spec, size_t image_half_size,
                                 double k, double b) {
    // FFT
    fftw_plan p = fftw_plan_dft_2d(h, w, reinterpret_cast<fftw_complex*>(image),reinterpret_cast<fftw_complex*>(image),
                                   FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    // Complex module
    #pragma omp parallel for
    for (int i = 0; i < h*w; i++)
    {
        image[i] = abs(image[i]);
    }

    // Angled spectrum
    #pragma omp parallel for
    for (int s = 0; s < image_half_size; s++)
    {
        // Init
        int points_count = 0;
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
                    points_count++;
                    av_ang_spec[s] += image[i*w + j].real();
                }
            }
        }

        // Averaged angled spectrum
        av_ang_spec[s] /= points_count;

        // Scaled arctan from averaged angled spectrum multiplied by linear function
        av_ang_spec[s] = atan(av_ang_spec[s] * (k * (s * M_PI / image_half_size) + b));
    }
}


/**
 * Evaluates defocus parameter restoration functional.
 * 
 * @param otf_slice OTF slice
 * @param norm normalized average angled spectrum of an image
 * @param norm_size norm size
 * @param norm_zeros indexes where norm zeros are located
 * @param zeros_count count of found zeros
 * @param zeros_range_ratio till which point we compair zeros
 * @param alpha first (functional's) sum coefficient
 * @param beta second (functional's) sum coefficient
 */
double compute_functional(const double *otf_slice, const double *norm, size_t norm_size, const int *norm_zeros,
                          size_t zeros_count, double zeros_range_ratio, double alpha, double beta) {
    // First sum in functional
    double otf_slice_sum = 0;
    // Sum of abs values in zero points of spectrum
    #pragma omp parallel for reduction(+:otf_slice_sum)
    for (int i = 0; i < zeros_count; i++)
    {
        otf_slice_sum += fabs(otf_slice[norm_zeros[i]]);
    }

    // Second sum in functional
    double norm_sum = 0;
    // Sum of abs values in zero points of spectrum
    #pragma omp parallel for reduction(+:norm_sum)
    for (int i = 1; i < int(zeros_range_ratio * (norm_size/M_PI)); i++)
    {
        // If we hit pure 0
        if (otf_slice[i] == 0)
        {
            // Add value
            norm_sum += norm[i];
        }
        // If there is a sign change
        else if ((otf_slice[i] > 0 && otf_slice[i-1] < 0) || (otf_slice[i] < 0 && otf_slice[i-1] > 0))
        {
            // Add half sum
            norm_sum += (norm[i-1] + norm[i]) / 2;
        }
    }

    // Functional formula
    return alpha * otf_slice_sum + beta * norm_sum;
}


double evaluate_fuctional(double a, const double *r, const double *norm, size_t norm_size, const int *norm_zeros,
                          size_t zeros_count, double zeros_range_ratio, double alpha, double beta) {
    // Allocate OTF slice memory
    double *otf_slice = (double*)malloc(sizeof(double) * norm_size);
    // Copy radius grid
    memcpy(otf_slice, r, sizeof(double) * norm_size);
    // Evaluate OTF
    evaluate_otf_on_grid(otf_slice, norm_size, a);

    // Compute functional value
    double value = compute_functional(otf_slice, norm, norm_size, norm_zeros, zeros_count, zeros_range_ratio, alpha, beta);

    // Free OTF slice
    free(otf_slice);

    return value;
}


void evaluate_fuctional_on_grid(double *a, size_t a_size, const double *r, const double *norm, size_t norm_size,
                                const int *norm_zeros, size_t zeros_count, double zeros_range_ratio, double alpha, double beta) {
    // Evaluate value at each grid point
    #pragma omp parallel for
    for (int i = 0; i < a_size; i++)
    {
        a[i] = evaluate_fuctional(a[i], r, norm, norm_size, norm_zeros, zeros_count, zeros_range_ratio, alpha, beta);
    }
}


double evaluate_fuctional_rgb(double a, double wlength0, const double *r, const double *norm, size_t norm_size, const int *norm_zeros,
                              const int *zeros_count, double zeros_range_ratio, const double* rgb_ratios,
                              const double *detector_funcs, size_t wlgrl, double alpha, double beta) {
    // Allocate OTF slice memory
    double *otf_batch = (double*)malloc(sizeof(double) * norm_size * 9);
    // Copy radius grid
    memcpy(otf_batch, r, sizeof(double) * norm_size);
    // Evaluate OTF
    evaluate_otf_batch(otf_batch, r, norm_size, rgb_ratios, detector_funcs, wlgrl, a, wlength0);

    // Combine OTF for each row
    for (int i = 0; i < 3; i++)
    {
        for (int p = 0; p < norm_size; p++)
        {
            otf_batch[(i*3)*norm_size + p] = (otf_batch[(i*3)*norm_size + p] + otf_batch[(i*3 + 1)*norm_size + p]
                                              + otf_batch[(i*3 + 2)*norm_size + p])/3;
        }
    }

    // Functional value
    double value = 0;

    // Compute functional value for each RGB channel
    value += compute_functional(otf_batch, norm, norm_size,
                                norm_zeros, zeros_count[0],
                                zeros_range_ratio, alpha, beta);
    value += compute_functional(otf_batch + 3*norm_size, norm + norm_size, norm_size,
                                norm_zeros+zeros_count[0], zeros_count[1],
                                zeros_range_ratio, alpha, beta);
    value += compute_functional(otf_batch + 6*norm_size, norm + 2*norm_size, norm_size,
                                norm_zeros+zeros_count[0]+zeros_count[1], zeros_count[2],
                                zeros_range_ratio, alpha, beta);

    // Free OTF slice
    free(otf_batch);

    return value;
}


void evaluate_fuctional_on_grid_rgb(double *functional_values, double *a, size_t a_size, double *wlength0, size_t wlength0_size,
                                    const double *r, const double *norm, size_t norm_size, const int *norm_zeros,
                                    const int *zeros_count, double zeros_range_ratio,
                                    const double* rgb_ratios, const double *detector_funcs, size_t wlgrl,
                                    double alpha, double beta) {
    // Evaluate value at each grid point
    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < a_size; i++)
    {
        for (int j = 0; j < wlength0_size; j++)
        {
            functional_values[i*wlength0_size + j] = evaluate_fuctional_rgb(a[i], wlength0[j], r, norm, norm_size,
                                                                            norm_zeros, zeros_count, zeros_range_ratio,
                                                                            rgb_ratios, detector_funcs, wlgrl, alpha, beta);
        }
    }
}