###########
# IMPORTS #
###########


# Handy arrays
import numpy as np
# Custom modules
import lib.utils.utils as utils


#########################
# Image Sample Presets  #
#########################


class SimulatedSamplePreset:
    """Stores info about simulated sample.
    """
    
    def __init__(self, image: np.ndarray, sample_folder: str, detector_funcs: np.ndarray,
                 average_angled_spectrum_k: float, average_angled_spectrum_b: float, zeros_range_ratio: float,
                 defocus_a_grid: float, focus_wlength_grid: float,
                 defocus_a: float, focus_wlength: float, additive_noise: np.ndarray,
                 mu: np.ndarray, k0: int):
        # Sample image path
        self.image = image
        # Image RGB ratios
        self.image_rgb_ratios = utils.image_rgb_max_via_histogram(image)
        # Where to put results
        self.sample_folder = sample_folder
        # Detector functions of optical system
        self.detector_funcs = detector_funcs.astype(np.float64)
        # k in calculation of average angled spectrum
        self.average_angled_spectrum_k = average_angled_spectrum_k
        # b in calculation of average angled spectrum
        self.average_angled_spectrum_b = average_angled_spectrum_b
        # Radius range in terms of pi, where zeros will be searched
        self.zeros_range_ratio = zeros_range_ratio
        # Defocus parameter grid for parameters restoration
        self.defocus_a_grid = defocus_a_grid
        # Focus wave length grid for parameters restoration
        self.focus_wlength_grid = focus_wlength_grid
        # Defocus parameter
        self.defocus_a = defocus_a
        # Focus wave length
        self.focus_wlength = focus_wlength
        # Additive noise
        self.additive_noise = additive_noise
        # Restoration method param
        self.mu = mu
        # Number of restoration method iterations
        self.k0 = k0


class SamplePreset:
    """Stores info about sample.
    """
    
    def __init__(self, image: np.ndarray, image_orig: np.ndarray, sample_folder: str, detector_funcs: np.ndarray,
                 average_angled_spectrum_k: float, average_angled_spectrum_b: float, zeros_range_ratio: float,
                 defocus_a_grid: float, focus_wlength_grid: float,
                 defocus_a: float, focus_wlength: float,
                 mu: np.ndarray, k0: int, max_iters: int):
        # Blurred image
        self.image = image
        # Original image
        self.image_orig = image_orig
        # Image RGB ratios
        self.image_rgb_ratios = utils.image_rgb_max_via_histogram(image)
        # Where to put results
        self.sample_folder = sample_folder
        # Detector functions of optical system
        self.detector_funcs = detector_funcs.astype(np.float64)
        # k in calculation of average angled spectrum
        self.average_angled_spectrum_k = average_angled_spectrum_k
        # b in calculation of average angled spectrum
        self.average_angled_spectrum_b = average_angled_spectrum_b
        # Radius range in terms of pi, where zeros will be searched
        self.zeros_range_ratio = zeros_range_ratio
        # Defocus parameter grid for parameters restoration
        self.defocus_a_grid = defocus_a_grid
        # Focus wave length grid for parameters restoration
        self.focus_wlength_grid = focus_wlength_grid
        # Defocus parameter
        self.defocus_a = defocus_a
        # Focus wave length
        self.focus_wlength = focus_wlength
        # Restoration method param
        self.mu = mu
        # Number of restoration method iterations
        self.k0 = k0
        # Max number of method iterations (when building discrepancy graph)
        self.max_iters = max_iters