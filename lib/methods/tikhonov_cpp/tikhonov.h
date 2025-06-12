// Used by dll
#ifdef DLL
  #define DLLEXPORT __declspec(dllexport)
#else
  #define DLLEXPORT __declspec(dllimport)
#endif

// Complex numbers
#include <complex>

/**
 * Tikhonov regularization method of image restoration.
 * 
 * @param image image to restore
 * @param h height
 * @param w width
 * @param otf OTF
 * @param alpha regularization param
 * @param r regularization param
**/
extern "C" DLLEXPORT void tikhonov_regularization_method(std::complex<double> *image, size_t h, size_t w, std::complex<double> const *otf,
                                                         double alpha, double r);