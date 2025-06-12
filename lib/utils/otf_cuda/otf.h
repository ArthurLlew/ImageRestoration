// Used by dll
#ifdef DLL
  #define DLLEXPORT __declspec(dllexport)
#else
  #define DLLEXPORT __declspec(dllimport)
#endif

// Complex numbers
#include <complex>

/**
 * PSF creation.
 * 
 * @param psf allocaded PSF memory
 * @param h PSF height
 * @param w PSF width
 * @param a defocus param
 * @param R pupil radius
*/
extern "C" DLLEXPORT void create_psf(std::complex<double> *psf, size_t h, size_t w, double a, double R);

/**
 * OTF creation.
 * 
 * @param otf allocaded OTF memory
 * @param h OTF height
 * @param w OTF width
 * @param a defocus param
 * @param R pupil radius
*/
extern "C" DLLEXPORT void create_otf(std::complex<double> *psf, size_t h, size_t w, double a, double R);

/**
 * OTF batch creation.
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
extern "C" DLLEXPORT void create_otf_batch(std::complex<double> *otf_bacth, size_t h, size_t w, const double* rgb_ratios,
                                           const double *detector_funcs, size_t wlgrl, double a, double wlength0, double R);