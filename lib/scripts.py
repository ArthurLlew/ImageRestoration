###########
# IMPORTS #
###########


# Math
import math
# Handy arrays
import numpy as np
# Extremums
from scipy.signal import argrelextrema
# Custom modules
import lib.utils.utils as utils
import lib.utils.plotting as plotting
import lib.utils.otf as OTF
import lib.utils.metrics as metrics
import lib.methods.otf_restoration as otf_restoration


###########
# Scripts #
###########


def estimate_defocus_param(image_blurred: np.ndarray, k: float, b: float, zeros_range_ratio: float, alpha=2.0, beta=1.0,
                           plot_folder='') -> float:
    """Estimates defocus param from given image using provided params.
    """

    utils.check_shape_is_2d(image_blurred.shape)
    
    # Radial spectrum averaging
    average_angled_spectrum = otf_restoration.average_angled_spectrum(image_blurred, k, b)
    # Radial grid
    r = np.arange(0, math.pi, math.pi/average_angled_spectrum.shape[0])
    # Envelope
    envelope = otf_restoration.get_lower_envelope(r, average_angled_spectrum)
    plotting.plot_functions([plotting.FuncDesc(r, average_angled_spectrum, r'$U_{a}(r)$'),
                             plotting.FuncDesc(r, envelope, r'$U_{env}(r)$')],
                            x_axis_name='Radius', plot_file=plot_folder+'3.svg')
    
    # Average angled spectrum normalized by envelope
    average_angled_spectrum_norm = otf_restoration.normalize_by_envelope(average_angled_spectrum, envelope)

    # Estimate zeros of the norm
    norm_zeros = otf_restoration.get_norm_zeros(average_angled_spectrum_norm, zeros_range_ratio)
    if norm_zeros.size == 0:
        raise Exception('No zeros found')
    
    # Plot zeros and norm
    norm_zeros_plot = np.zeros(norm_zeros.shape)
    plotting.plot_functions([plotting.FuncDesc(r, average_angled_spectrum_norm, 'U(r)'),
                             plotting.FuncDesc(r[norm_zeros], norm_zeros_plot, 'Zeros', 'o')],
                            x_axis_name='Radius', plot_file=plot_folder+'4.svg')

    # Defocus param grid
    a = np.arange(0.1, 20, 0.1)
    # Functional values calculation
    functional_values = otf_restoration.evaluate_fuctional(a, r, average_angled_spectrum_norm, norm_zeros, zeros_range_ratio,
                                                           alpha=alpha, beta=beta)
    plotting.plot_functions([plotting.FuncDesc(a, functional_values, 'F(a)')],
                            x_axis_name='Defocus parameter', plot_file=plot_folder+'5.svg')
    
    # Minimization
    def minimise_functional(functional_values, segment_splitting):
        return segment_splitting[np.argmin(functional_values)]
    restored_a = minimise_functional(functional_values, a)
    
    # Show results of parameter restoration
    print('Min defocus param is: ', restored_a)
    print('Other known mins:')
    print(a[argrelextrema(functional_values, np.less)])
    
    return restored_a


def estimate_defocus_param_rgb(image_blurred: np.ndarray, k: float, b: float, zeros_range_ratio: float,
                               image_rgb_ratios: np.ndarray, detector_funcs: np.ndarray, alpha=2.0, beta=1.0,
                               plot_folder='', a=None, wlength0=None) -> float:
    """Estimates defocus param from given image using provided params.
    """

    utils.check_shape_is_rgb(image_blurred.shape)

    average_angled_spectrum_norm = [0, 0, 0]
    norm_zeros = [0, 0, 0]
    
    # Analyze all RGB image channels
    for i in range(3):
        # Radial spectrum averaging
        average_angled_spectrum = otf_restoration.average_angled_spectrum(image_blurred[:, :, i], k, b)
        # Radial grid
        r = np.arange(0, math.pi, math.pi/average_angled_spectrum.shape[0])
        # Envelope
        envelope = otf_restoration.get_lower_envelope(r, average_angled_spectrum)
        plotting.plot_functions([plotting.FuncDesc(r, average_angled_spectrum, r'$U_{a}(r)$'),
                                 plotting.FuncDesc(r, envelope, r'$U_{env}(r)$')],
                                x_axis_name='Radius', plot_file=plot_folder+'4-1-'+str(i)+'.svg')
        
        # Average angled spectrum normalized by envelope
        average_angled_spectrum_norm[i] = otf_restoration.normalize_by_envelope(average_angled_spectrum, envelope)

        # Estimate zeros of the norm
        norm_zeros[i] = otf_restoration.get_norm_zeros(average_angled_spectrum_norm[i], zeros_range_ratio)
        if norm_zeros[i].size == 0:
            raise Exception('No zeros found')
        
        # Plot zeros and norm
        norm_zeros_plot = np.zeros(norm_zeros[i].shape)
        plotting.plot_functions([plotting.FuncDesc(r, average_angled_spectrum_norm[i], 'U(r)'),
                                 plotting.FuncDesc(r[norm_zeros[i]], norm_zeros_plot, 'Zeros', 'o')],
                                x_axis_name='Radius', plot_file=plot_folder+'4-2-'+str(i)+'.svg')
        
    # Defocus param grid
    if a is None:
        a = np.arange(0.001, 0.1, 0.002).astype(np.float64)
    if wlength0 is None:
        wlength0 = np.arange(855, 955, 2).astype(np.float64)
    # Functional values calculation
    functional_values = otf_restoration.evaluate_fuctional_rgb(a, wlength0, r, average_angled_spectrum_norm, norm_zeros,
                                                               zeros_range_ratio, image_rgb_ratios, detector_funcs,
                                                               alpha=alpha, beta=beta)

    # Plot logarithm scaled functional values
    plotting.plot_colormesh(wlength0, a, np.log(1 + functional_values), plot_file=plot_folder+'4-3.svg')
    
    return functional_values


def compair_results(sample_folder: str) -> None:
    """Compairs original and restored images via metrics.
    """
    
    # Load precomputed images
    image_original = utils.image_load(sample_folder + '1, original.png')
    image_restored_deconv1 = utils.image_load(sample_folder + '8, blind deconv(1).png')
    image_restored_deconv2 = utils.image_load(sample_folder + '8, blind deconv(2).png')
    image_restored_bisp = utils.image_load(sample_folder + '6, bisp.png')
    image_restored_tich = utils.image_load(sample_folder + '7, tich.png')
    
    print('Blind deconvolution 1 metrics:')
    print('MSE %4f' % metrics.compare_images_mse(image_original, image_restored_deconv1))
    print('SSIM %4f' % metrics.compare_images_ssim(image_original, image_restored_deconv1))

    print('Blind deconvolution 2 metrics:')
    print('MSE %4f' % metrics.compare_images_mse(image_original, image_restored_deconv2))
    print('SSIM %4f' % metrics.compare_images_ssim(image_original, image_restored_deconv2))
    
    print('Bispectral method metrics:')
    print('MSE %4f' % metrics.compare_images_mse(image_original, image_restored_bisp))
    print('SSIM %4f' % metrics.compare_images_ssim(image_original, image_restored_bisp))

    print('Tikhonov method metrics:')
    print('MSE %4f' % metrics.compare_images_mse(image_original, image_restored_tich))
    print('SSIM %4f' % metrics.compare_images_ssim(image_original, image_restored_tich))