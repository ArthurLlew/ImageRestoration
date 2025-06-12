###########
# IMPORTS #
###########


# OS
import os
# Used to manage DLLs
import ctypes as ct
# Handy arrays
import numpy as np
# FFTs
import scipy.fft as fft
# Custom modules
import lib.utils.utils as utils


################
# DLL Handling #
################


# Load DLL
dll_restoration_cuda = ct.CDLL(os.path.join(os.path.dirname(__file__), 'tikhonov_rgb_cuda', 'tikhonov_rgb'))

# Describe DLL functions:
# Tikhonov regularization method of RGB image restoration (CUDA)
dll_restoration_cuda.tikhonov_regularization_method_rgb.argtypes = (np.ctypeslib.ndpointer(dtype=np.complex128, ndim=3),
                                                                    ct.c_size_t, ct.c_size_t,
                                                                    np.ctypeslib.ndpointer(dtype=np.complex128, ndim=4),
                                                                    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2), ct.c_int,
                                                                    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1))
dll_restoration_cuda.tikhonov_regularization_method_rgb.restype = None


######################
# Restoration method #
######################


def tikhonov_regularization_method_rgb(image: np.ndarray, otf_batch: np.ndarray, mu: None | list[float, float] | np.ndarray,
                                       k0: int) -> tuple[np.ndarray,np.ndarray]:
    """Iterative tikhonov regularization method of RGB image restoration.
    """

    # Check mu
    if mu is None:
        # Use formula with defaults
        mu = get_regularization_param(image)
    elif isinstance(mu, list):
        # Use formula with provided parameters
        mu = get_regularization_param(image, min_adjustment=mu[0], scale=mu[1])
    elif not isinstance(mu, np.ndarray):
        raise ValueError('Incorrect mu type!')

    # Copy observed image
    image_restored = np.copy(image).astype(np.complex128)

    # Change shape from (3,3,h,w) to (h,w,3,3) and make a copy (to retain shape outside of this function)
    otf_batch_ = np.moveaxis(otf_batch, (0, 1, 2, 3), (2, 3, 0, 1)).astype(np.complex128)

    # Discrepancy
    discrepancy = np.zeros(abs(k0)).astype(np.float64)

    # Make array C contiguous
    if not otf_batch_.flags['C_CONTIGUOUS']:
        otf_batch_ = np.ascontiguousarray(otf_batch_)
    if not image_restored.flags['C_CONTIGUOUS']:
        image_restored = np.ascontiguousarray(image_restored)
    if not mu.flags['C_CONTIGUOUS']:
        mu = np.ascontiguousarray(mu)
    if not discrepancy.flags['C_CONTIGUOUS']:
        discrepancy = np.ascontiguousarray(discrepancy)

    # Call DLL function
    dll_restoration_cuda.tikhonov_regularization_method_rgb(image_restored, *image_restored.shape[0:2],
                                                            otf_batch_, mu, ct.c_int(k0), discrepancy)

    # Return normalized and autocontrasted real part
    return utils.image_autocontrast(utils.image_normalize(image_restored.real)), discrepancy


def get_regularization_param(image: np.ndarray, min_adjustment=0.01, scale=0.25) -> np.ndarray:
    """Produces regularization parameter via energy spectrum.
    """

    # Energy spectrum
    mu = utils.get_spectrum(fft.fft2(utils.rgb2grayscale(image)))

    # Scaling + correct numpy type
    return ((mu.max() - mu + min_adjustment) * scale).astype(np.float64)