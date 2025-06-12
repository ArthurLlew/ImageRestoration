###########
# IMPORTS #
###########


# Handy arrays
import numpy as np


########################
# Image Sample Presets #
########################


class SimulatedSamplePreset:
    """Stores info about simulated sample.
    """
    
    def __init__(self, filepath: str, crop_slice: tuple[slice, slice], sample_folder: str,
                 defocus_param: float, additive_noise: np.ndarray,
                 average_angled_spectrum_k: float, average_angled_spectrum_b: float, zeros_range_ratio: float,
                 deconv_iters: int,
                 bispectral_alpha: float,
                 tikhonov_alpha: float, tikhonov_r: float,
                 crop_ratio: float):
        # Sample image path
        self.filepath = filepath
        # How to slice original image
        self.crop_slice = crop_slice
        # Where to put results
        self.sample_folder = sample_folder
        # Defocus param to use
        self.defocus_param = defocus_param
        # Additive noise
        self.additive_noise = additive_noise
        # k in calculation of average angled spectrum
        self.average_angled_spectrum_k = average_angled_spectrum_k
        # b in calculation of average angled spectrum
        self.average_angled_spectrum_b = average_angled_spectrum_b
        # Radius range in terms of pi, where zeros will be searched
        self.zeros_range_ratio = zeros_range_ratio
        # Number of blind deconvolution iterations
        self.deconv_iters = deconv_iters
        # alpha in bispectral method
        self.bispectral_alpha = bispectral_alpha
        # alpha in tikhonov method
        self.tikhonov_alpha = tikhonov_alpha
        # r in tikhonov method
        self.tikhonov_r = tikhonov_r
        # Cropping ratio for saved images
        self.crop_ratio = crop_ratio


class SamplePreset:
    """Stores info about sample.
    """
    
    def __init__(self, filepath: str, filepath_orig: str, crop_slice: tuple[slice, slice], sample_folder: str,
                 average_angled_spectrum_k: float, average_angled_spectrum_b: float, zeros_range_ratio: float,
                 defocus_param: float,
                 deconv_iters: int,
                 bispectral_alpha: float,
                 tikhonov_alpha: float, tikhonov_r: float,
                 crop_ratio: float):
        # Sample image path
        self.filepath = filepath
        # Original image path
        self.filepath_orig = filepath_orig
        # How to slice original image
        self.crop_slice = crop_slice
        # Where to put results
        self.sample_folder = sample_folder
        # k in calculation of average angled spectrum
        self.average_angled_spectrum_k = average_angled_spectrum_k
        # b in calculation of average angled spectrum
        self.average_angled_spectrum_b = average_angled_spectrum_b
        # Radius range in terms of pi, where zeros will be searched
        self.zeros_range_ratio = zeros_range_ratio
        # Restored defocus param
        self.defocus_param = defocus_param
        # Number of blind deconvolution iterations
        self.deconv_iters = deconv_iters
        # alpha in bispectral method
        self.bispectral_alpha = bispectral_alpha
        # alpha in tikhonov method
        self.tikhonov_alpha = tikhonov_alpha
        # r in tikhonov method
        self.tikhonov_r = tikhonov_r
        # Cropping ratio for saved images
        self.crop_ratio = crop_ratio