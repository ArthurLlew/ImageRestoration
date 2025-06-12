// Used by dll
#ifdef DLL
  #define DLLEXPORT __declspec(dllexport)
#else
  #define DLLEXPORT __declspec(dllimport)
#endif

// Complex numbers
#include <complex>

/**
 * Calculates average angled spectrum from given image.
 * 
 * @param image image to analyze
 * @param h image height
 * @param w image width
 * @param av_ang_spec average angled spectrum
 * @param image_half_size average angled spectrum size
 * @param k k * x
 * @param b + b
**/
extern "C" DLLEXPORT void get_average_angled_spectrum(std::complex<double> *image, size_t h, size_t w, double *av_ang_spec,
                                                      size_t image_half_size, double k, double b);

/**
 * Evaluates defocus parameter restoration functional.
 * 
 * @param a defocus param
 * @param r radius grid
 * @param norm normalized average angled spectrum of an image
 * @param norm_size norm size
 * @param norm_zeros indexes where norm zeros are located
 * @param zeros_count count of found zeros
 * @param zeros_range_ratio till which point we compair zeros
 * @param alpha first (functional's) sum coefficient
 * @param beta second (functional's) sum coefficient
 */
extern "C" DLLEXPORT double evaluate_fuctional(double a, const double *r, const double *norm, size_t norm_size, const int *norm_zeros,
                                               size_t zeros_count, double zeros_range_ratio, double alpha, double beta);

/**
 * Evaluates defocus parameter restoration functional.
 * 
 * @param a defocus param grid
 * @param a_size defocus param grid size
 * @param r radius grid
 * @param norm normalized average angled spectrum of an image
 * @param norm_size norm size
 * @param norm_zeros indexes where norm zeros are located
 * @param zeros_count count of found zeros
 * @param zeros_range_ratio till which point we compair zeros
 * @param alpha first (functional's) sum coefficient
 * @param beta second (functional's) sum coefficient
 */
extern "C" DLLEXPORT void evaluate_fuctional_on_grid(double *a, size_t a_size, const double *r, const double *norm, size_t norm_size,
                                                     const int *norm_zeros, size_t zeros_count, double zeros_range_ratio,
                                                     double alpha, double beta);

/**
 * Evaluates defocus parameter restoration functional.
 * 
 * @param a defocus param
 * @param wlength0 focus wave length
 * @param r radius grid
 * @param norm normalized average angled spectrum of an image
 * @param norm_size norm size
 * @param norm_zeros indexes where norm zeros are located
 * @param zeros_count count of found zeros
 * @param zeros_range_ratio till which point we compair zeros
 * @param rgb_ratios RGB channel ratios that will define the image color
 * @param detector_funcs Wave length grid and detector functions of RGB channels
 * @param wlgrl Wave length grid length
 * @param alpha first (functional's) sum coefficient
 * @param beta second (functional's) sum coefficient
 */
extern "C" DLLEXPORT double evaluate_fuctional_rgb(double a, double wlength0, const double *r, const double *norm,
                                                   size_t norm_size, const int *norm_zeros, const int *zeros_count,
                                                   double zeros_range_ratio, const double* rgb_ratios,
                                                   const double *detector_funcs, size_t wlgrl, double alpha, double beta);

/**
 * Evaluates defocus parameter restoration functional.
 * 
 * @param functional_values functional values array
 * @param a defocus param grid
 * @param a_size defocus param grid size
 * @param wlength0 focus wave length grid
 * @param wlength0_size focus wave length grid size
 * @param r radius grid
 * @param norm normalized average angled spectrum of an image
 * @param norm_size norm size
 * @param norm_zeros indexes where norm zeros are located
 * @param zeros_count count of found zeros
 * @param zeros_range_ratio till which point we compair zeros
 * @param rgb_ratios RGB channel ratios that will define the image color
 * @param detector_funcs Wave length grid and detector functions of RGB channels
 * @param wlgrl Wave length grid length
 * @param alpha first (functional's) sum coefficient
 * @param beta second (functional's) sum coefficient
 */
extern "C" DLLEXPORT void evaluate_fuctional_on_grid_rgb(double *functional_values, double *a, size_t a_size,
                                                         double *wlength0, size_t wlength0_size,
                                                         const double *r, const double *norm, size_t norm_size,
                                                         const int *norm_zeros, const int *zeros_count,
                                                         double zeros_range_ratio, const double* rgb_ratios,
                                                         const double *detector_funcs, size_t wlgrl,
                                                         double alpha, double beta);