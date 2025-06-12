// Used by dll
#ifdef DLL
  #define DLLEXPORT __declspec(dllexport)
#else
  #define DLLEXPORT __declspec(dllimport)
#endif

// Complex numbers
#include <complex>

/**
 * Bispectral method of image restoration.
 * 
 * @param image image to restore
 * @param h height
 * @param w width
 * @param otf OTF
 * @param alpha param
 * @param image_restored FT of the restored image
**/
extern "C" DLLEXPORT void bispectral_method(std::complex<double> *image, size_t h, size_t w,
                                            std::complex<double> const *otf, double alpha);