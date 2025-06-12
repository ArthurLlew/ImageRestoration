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
*/
extern "C" DLLEXPORT void get_average_angled_spectrum(std::complex<double> *image, size_t h, size_t w, double *av_ang_spec,
                                                      size_t image_half_size, double k, double b);