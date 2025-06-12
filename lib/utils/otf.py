###########
# IMPORTS #
###########


# Math
import math
# OS
import os
# Used to manage DLLs
import ctypes as ct
# Handy arrays
import numpy as np
# Integration
import scipy
import scipy.integrate


################
# DLL Handling #
################


# Load DLL
dll_otf_cpp = ct.CDLL(os.path.join(os.path.dirname(__file__), 'otf_cpp', 'otf'))
dll_otf_cuda = ct.CDLL(os.path.join(os.path.dirname(__file__), 'otf_cuda', 'otf'))

# Describe DLL functions:
# Create PSF (C++)
dll_otf_cpp.create_psf.argtypes = (np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2), ct.c_size_t, ct.c_size_t,
                                   ct.c_double, ct.c_double)
dll_otf_cpp.create_psf.restype = None
# PSF to OTF (C++)
dll_otf_cpp.psf2otf.argtypes = (np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2), ct.c_size_t, ct.c_size_t)
dll_otf_cpp.psf2otf.restype = None
# Create OTF (C++)
dll_otf_cpp.create_otf.argtypes = (np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2), ct.c_size_t, ct.c_size_t,
                                   ct.c_double, ct.c_double)
dll_otf_cpp.create_otf.restype = None
# Create PSF (CUDA)
dll_otf_cuda.create_psf.argtypes = (np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2), ct.c_size_t, ct.c_size_t,
                                    ct.c_double, ct.c_double)
dll_otf_cuda.create_psf.restype = None
# Create OTF (CUDA)
dll_otf_cuda.create_otf.argtypes = (np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2), ct.c_size_t, ct.c_size_t,
                                    ct.c_double, ct.c_double)
dll_otf_cuda.create_otf.restype = None
# Create OTF batch (CUDA)
dll_otf_cuda.create_otf_batch.argtypes = (np.ctypeslib.ndpointer(dtype=np.complex128, ndim=4), ct.c_size_t, ct.c_size_t,
                                          np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                          np.ctypeslib.ndpointer(dtype=np.float64, ndim=2), ct.c_size_t,
                                          ct.c_double, ct.c_double, ct.c_double)
dll_otf_cuda.create_otf_batch.restype = None
# OTF fast evaluation in point (C++)
dll_otf_cpp.evaluate_otf_in_point.argtypes = (ct.c_double, ct.c_double)
dll_otf_cpp.evaluate_otf_in_point.restype = ct.c_double
# OTF fast evaluation over grid (C++)
dll_otf_cpp.evaluate_otf_on_grid.argtypes = (np.ctypeslib.ndpointer(dtype=np.float64, ndim=1), ct.c_size_t, ct.c_double)
dll_otf_cpp.evaluate_otf_on_grid.restype = None
# OTF batch fast evaluation (C++)
dll_otf_cpp.evaluate_otf_batch.argtypes = (np.ctypeslib.ndpointer(dtype=np.float64, ndim=3),
                                           np.ctypeslib.ndpointer(dtype=np.float64, ndim=1), ct.c_size_t,
                                           np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                           np.ctypeslib.ndpointer(dtype=np.float64, ndim=2), ct.c_size_t,
                                           ct.c_double, ct.c_double)
dll_otf_cpp.evaluate_otf_batch.restype = None


###########
# PSF/OTF #
###########


def create_psf(shape: tuple, a: float, R=0.5) -> np.ndarray:
    """Creates PSF.
    """

    # Init psf memory
    psf = np.empty(shape, dtype=np.complex128)
    # Make array C contiguous
    if not psf.flags['C_CONTIGUOUS']:
        psf = np.ascontiguousarray(psf)

    # Call DLL function (depending on image size)
    if shape[0] < 2048 and shape[1] < 2048:
        dll_otf_cpp.create_psf(psf, *psf.shape, ct.c_double(a), ct.c_double(R))
    else:
        dll_otf_cuda.create_psf(psf, *psf.shape, ct.c_double(a), ct.c_double(R))

    # Return filled memory
    return psf.real


def psf2otf(psf: np.ndarray) -> np.ndarray:
    """Converts PSF to OTF.
    """

    # Make complex copy
    otf = psf.astype(np.complex128)
    # Make array C contiguous
    if not otf.flags['C_CONTIGUOUS']:
        otf = np.ascontiguousarray(otf)

    # Call DLL function 
    dll_otf_cpp.psf2otf(otf, *otf.shape)

    return otf


def create_otf(shape: tuple, a: float, R=0.5) -> np.ndarray:
    """Creates OTF.
    """

    # Init otf memory
    otf = np.empty(shape, dtype=np.complex128)
    # Make array C contiguous
    if not otf.flags['C_CONTIGUOUS']:
        otf = np.ascontiguousarray(otf)

    # Call DLL function (depending on image size)
    if shape[0] < 2048 and shape[1] < 2048:
        dll_otf_cpp.create_psf(otf, *otf.shape, ct.c_double(a), ct.c_double(R))
        dll_otf_cpp.psf2otf(otf, *otf.shape)
    else:
        dll_otf_cuda.create_otf(otf, *otf.shape, ct.c_double(a), ct.c_double(R))

    # Return filled memory
    return otf


#######################
# PSF/OTF in RGB case #
#######################


def create_otf_batch(shape: tuple, rgb_ratios: np.ndarray, detector_funcs: np.ndarray, a: float, wlength0: float, R=0.5) -> np.ndarray:
    """Produces 3x3 OTF batch according to provided RGB detector functions, focus wavelength and defocus parameter.
    """

    # Init result
    otf_batch = np.empty((3, 3, shape[0], shape[1]), dtype=np.complex128)

    # Make arrays C contiguous
    if not detector_funcs.flags['C_CONTIGUOUS']:
        detector_funcs = np.ascontiguousarray(detector_funcs)
    if not otf_batch.flags['C_CONTIGUOUS']:
        otf_batch = np.ascontiguousarray(otf_batch)
    if not rgb_ratios.flags['C_CONTIGUOUS']:
        rgb_ratios = np.ascontiguousarray(rgb_ratios)

    # Call DLL function
    dll_otf_cuda.create_otf_batch(otf_batch, *shape[0:2], rgb_ratios, detector_funcs, detector_funcs.shape[1],
                                  ct.c_double(a), ct.c_double(wlength0), ct.c_double(R))

    return otf_batch


#############################
# OTF Fast Evaluation Utils #
#############################


def evaluate_otf_in_point(r: float, a: float) -> float:
    """Evaluates OTF radial real part with given defocus parameter at given radius.
    """

    # Check if radius is valid
    if (r < 0 or r > math.pi):
        raise ValueError("Radius is out of possible range [0, PI]")

    # Call DLL function
    return dll_otf_cpp.evaluate_otf_in_point(ct.c_double(r), ct.c_double(a))


def evaluate_otf_on_grid(r_grid: np.ndarray, a: float) -> np.ndarray:
    """Evaluates OTF radial real part with given defocus parameter over given radius grid.
    """

    # Init result
    otf = np.copy(r_grid)

    # Make arrays C contiguous
    if not otf.flags['C_CONTIGUOUS']:
        otf = np.ascontiguousarray(otf)

    # Call DLL function
    dll_otf_cpp.evaluate_otf_on_grid(otf, otf.shape[0], ct.c_double(a))

    return otf


def evaluate_otf_batch(r_grid: np.ndarray, rgb_ratios: np.ndarray, detector_funcs: np.ndarray, a: float, wlength0) -> np.ndarray:
    """Evaluates OTF radial real part with given defocus parameter over given radius grid.
    """

    # Init result
    otf_batch = np.empty((3, 3, r_grid.shape[0]))

    # Make arrays C contiguous
    if not otf_batch.flags['C_CONTIGUOUS']:
        otf_batch = np.ascontiguousarray(otf_batch)
    if not r_grid.flags['C_CONTIGUOUS']:
        r_grid = np.ascontiguousarray(r_grid)
    if not rgb_ratios.flags['C_CONTIGUOUS']:
        rgb_ratios = np.ascontiguousarray(rgb_ratios)
    if not detector_funcs.flags['C_CONTIGUOUS']:
        detector_funcs = np.ascontiguousarray(detector_funcs)

    # Call DLL function
    dll_otf_cpp.evaluate_otf_batch(otf_batch, r_grid, r_grid.shape[0], rgb_ratios, detector_funcs, detector_funcs.shape[1],
                                   ct.c_double(a), ct.c_double(wlength0))

    return otf_batch