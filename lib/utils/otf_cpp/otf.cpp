// For PI constant
#define _USE_MATH_DEFINES
// Header for this file
#include "otf.h"
// OpenMP
#include <omp.h>
// FFTW
#include <fftw3.h>
// GSL (for integration)
#include <gsl/gsl_integration.h>


void create_psf(std::complex<double> *psf, size_t h, size_t w, double a, double R)
{
    // Shift R to fit [-pi, pi] and save its square
    double R2 = (R * M_PI) * (R * M_PI);

    // Fill values of Pupil multiplied by a complex exponent
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            // "Radius" of the point (x, y) mapped to [-pi, pi]
            double x = ((double)j/(w/2) - 1) * M_PI;
            double y = (1 - (double)i/(h/2)) * M_PI;
            double r = x*x + y*y;
            // Pupil
            if (r > R2)
            {
                psf[i * w + j] = 0;
            }
            else
            {
                // Values inside pupil
                psf[i * w + j] = std::exp(std::complex<double>(0, a*r));
            }
        }
    }

    // FFT
    fftw_plan p = fftw_plan_dft_2d(h, w, reinterpret_cast<fftw_complex*>(psf), reinterpret_cast<fftw_complex*>(psf),
                                   FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    std::complex<double> size = std::complex<double>((h*w), 0);

    // Squared complex module (with psf divided by matrix size)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            psf[i * w + j] /= size;
            psf[i * w + j] *= std::conj(psf[i * w + j]);
        }
    }
}


void psf2otf(std::complex<double> *psf, size_t h, size_t w)
{
    // FFT
    fftw_plan p = fftw_plan_dft_2d(h, w, reinterpret_cast<fftw_complex*>(psf), reinterpret_cast<fftw_complex*>(psf),
                                   FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    size_t size = h*w;

    // According to formula
    std::complex<double> max_val = psf[0];
    // Divide by max value
    #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        psf[i] /= max_val;
    }
}


void create_otf(std::complex<double> *otf, size_t h, size_t w, double a, double R)
{
    // Create PSF
    create_psf(otf, h, w, a, R);
    // Convert it to OTF
    psf2otf(otf, h, w);
}


// Struct to hold integrated function parameters
struct otf_function_params
{
    double r; // radius
    double a; // defocus param
};


/**
 * OTF function under integral in fast evaluation formula
 */
double otf_function(double x, void * p)
{
    // Parse params
    struct otf_function_params * params = (struct otf_function_params *)p;
    double r = (params->r);
    double a = (params->a);

    // Compute function
    return sin(((4 * a * r) / M_PI) * (sqrt(1 - x*x) - r / M_PI)) / (a * r);
};


double evaluate_otf_in_point(double r, double a)
{
    // Check for zero argument (otherwise it will cause division by zero)
    if (r == 0.0)
    {
        // We know that OTF value at zero should be exactly 1
        return 1.0;
    }

    // Init GSL integration workspace
    gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(50);

    // Init GSL function to integration (works somewhat like lambda expression)
    struct otf_function_params params = { r, a };
    gsl_function F;
    F.function = &otf_function;
    F.params = &params;

    // Integration results
    double result;
    double abserr;
    // Integration like in Python's SciPy module
    gsl_integration_qags(&F, 0, sqrt(1 - (r * r) / (M_PI * M_PI)), 1.49e-8, 1.49e-8, 50, workspace, &result, &abserr);

    // Free memory
    gsl_integration_workspace_free(workspace);

    // Return integration result
    return result;
}


void evaluate_otf_on_grid(double *otf_real_radial, size_t size, double a)
{
    // Shift defocus param, so it will fit [-pi,pi] range
    double shifted_a = (M_PI / 2) * (M_PI / 2) * a;

    // Compute OTF real radial value at each point
    #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        otf_real_radial[i] = evaluate_otf_in_point(otf_real_radial[i], shifted_a);
    }
}


void evaluate_otf_batch(double *otf_batch, const double *r, size_t size, const double* rgb_ratios,
                        const double *detector_funcs, size_t wlgrl, double a, double wlength0) {
    // Allocate OTF grid memory
    double *otf_grid = (double*)malloc(sizeof(double) * size * wlgrl);

    // Compute OTF real radial part for each wave length in wave length grid with defocus param
    // according to formula
    #pragma omp parallel for
    for (int wl = 0; wl < wlgrl; wl++)
    {
        // Copy radius grid
        memcpy(otf_grid + size * wl, r, sizeof(double) * size);
        // Evaluate OTF
        evaluate_otf_on_grid(otf_grid + size * wl, size, a * (detector_funcs[wl] - wlength0));
    }

    // Wave length grid step
    double wlgstep = detector_funcs[1] - detector_funcs[0];

    // Compute OTF real radial part batch fro RGB case
    #pragma omp parallel for collapse(2)
    // Loop i,j over 3x3 batch
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            // Loop over OTF points
            for (int p = 0; p < size; p++)
            {
                // Clear value
                otf_batch[(i*3 + j)*size + p] = 0;

                // Loop over wave length grid
                for (int wl = 0; wl < wlgrl; wl++)
                {
                    // Inside the wave length grid
                    if ((wl != 0) && (wl != wlgrl-1))
                    {
                        // Step multiplied by function value
                        otf_batch[(i*3 + j)*size + p] += otf_grid[size * wl + p] * wlgstep
                                                          * detector_funcs[(1+i)*wlgrl + wl] * detector_funcs[(1+j)*wlgrl + wl];
                    }
                    // On the bounds of the wave length grid
                    else 
                    {
                        // Step, divided by 2, multiplied by function value
                        otf_batch[(i*3 + j)*size + p] += otf_grid[size * wl + p] * (wlgstep/2)
                                                          * detector_funcs[(1+i)*wlgrl + wl] * detector_funcs[(1+j)*wlgrl + wl];
                    }
                }
            }
                
            // According to formula
            double max_val = otf_batch[(i*3 + j)*size];
            for (int p = 0; p < size; p++)
            {
                // Divide by max value
                otf_batch[(i*3 + j)*size + p] /= max_val;
                // And multiply by RGB ratios
                otf_batch[(i*3 + j)*size + p] *= rgb_ratios[i] * rgb_ratios[j];
            }
        }
    }

    // Free OTF grid
    free(otf_grid);
}