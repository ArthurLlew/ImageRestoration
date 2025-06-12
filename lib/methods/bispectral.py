###########
# IMPORTS #
###########


# OS
import os
# Used to manage DLLs
import ctypes as ct
# Handy arrays
import numpy as np
# Custom modules
import lib.utils.utils as utils


################
# DLL Handling #
################


# Load DLL
dll_bispectral_cpp = ct.CDLL(os.path.join(os.path.dirname(__file__), 'bispectral_cpp', 'bispectral'))
# Bispectral Method
dll_bispectral_cpp.bispectral_method.argtypes = (np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2), ct.c_size_t, ct.c_size_t,
                                                 np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2), ct.c_double)
dll_bispectral_cpp.bispectral_method.restype = None


#################################
# Bispectral Restoration Method #
#################################


def bispectral_method(image: np.ndarray, otf: np.ndarray, alpha: float) -> np.ndarray:
    """Bispectral method of image restoration.
    """

    # Copy observed image
    image_restored = np.copy(image).astype(np.complex128)

    # Make array C contiguous
    if not image_restored.flags['C_CONTIGUOUS']:
        image_restored = np.ascontiguousarray(image_restored)
    if not otf.flags['C_CONTIGUOUS']:
        otf = np.ascontiguousarray(otf)

    # Call DLL function
    dll_bispectral_cpp.bispectral_method(image_restored, *image.shape[0:2], otf, ct.c_double(alpha))

    # Return normalized and autocontrasted real part
    return utils.image_autocontrast(utils.image_normalize(image_restored.real), threshold=0.003)