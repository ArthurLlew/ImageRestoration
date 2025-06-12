// Used by dll
#ifdef DLL
  #define DLLEXPORT __declspec(dllexport)
#else
  #define DLLEXPORT __declspec(dllimport)
#endif

// Complex numbers
#include <complex>

/**
 * Creates PSF.
 * 
 * @param psf allocaded PSF memory
 * @param h PSF height
 * @param w PSF width
 * @param a defocus param
 * @param R pupil radius
**/
extern "C" DLLEXPORT void create_psf(std::complex<double> *psf, size_t h, size_t w, double a, double R);

/**
 * Converts PSF to OTF.
 * 
 * @param psf allocaded PSF memory
 * @param h PSF height
 * @param w PSF width
**/
extern "C" DLLEXPORT void psf2otf(std::complex<double> *psf, size_t h, size_t w);

/**
 * Creates OTF.
 * 
 * @param otf allocaded OTF memory
 * @param h PSF height
 * @param w PSF width
 * @param a defocus param
 * @param R pupil radius
**/
extern "C" DLLEXPORT void create_otf(std::complex<double> *otf, size_t h, size_t w, double a, double R);


/**
 * Evaluates OTF radial real part in point.
 * 
 * @param r radius
 * @param a defocus param
**/
extern "C" DLLEXPORT double evaluate_otf_in_point(double r, double a);

/**
 * Evaluates OTF radial real part over given radius grid.
 * 
 * @param otf_real_radial allocaded OTF memory filled with radius values (basically radius grid)
 * @param size OTF size
 * @param a defocus param
**/
extern "C" DLLEXPORT void evaluate_otf_on_grid(double *otf_real_radial, size_t size, double a);


/**
 * Evaluates OTF radial real part RGB batch.
 * 
 * @param otf_batch OTF RGB batch
 * @param r radius grid
 * @param size radius grid size
 * @param rgb_ratios RGB channel ratios that will define the image color
 * @param detector_funcs Wave length grid and detector functions of RGB channels
 * @param wlgrl Wave length grid length
 * @param a Defocus param
 * @param wlength0 Focus wave length
**/
extern "C" DLLEXPORT void evaluate_otf_batch(double *otf_batch, const double *r, size_t size, const double* rgb_ratios,
                                             const double *detector_funcs, size_t wlgrl, double a, double wlength0);