// Used by dll
#ifdef DLL
  #define DLLEXPORT __declspec(dllexport)
#else
  #define DLLEXPORT __declspec(dllimport)
#endif

// Complex numbers
#include <complex>

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
extern "C" DLLEXPORT void tikhonov_regularization_method_rgb(std::complex<double> *image, size_t h, size_t w,
                                                             const std::complex<double> *otf_batch,
                                                             const double *mu, int k0, double *discrepancy);