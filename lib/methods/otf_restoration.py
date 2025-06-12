###########
# IMPORTS #
###########


# OS
import os
# Math
import math
# Handy arrays
import numpy as np
# Used to manage DLLs
import ctypes as ct
# Convex hulls
from scipy.spatial import ConvexHull
# Extremums
from scipy.signal import argrelextrema


################
# DLL Handling #
################


# Load DLL
dll_otf_restoration_cpp = ct.CDLL(os.path.join(os.path.dirname(__file__), 'otf_restoration_cpp', 'otf_restoration'))
dll_otf_restoration_cuda = ct.CDLL(os.path.join(os.path.dirname(__file__), 'otf_restoration_cuda', 'otf_restoration'))

# Describe DLL functions:
# Get average angled spectrum (C++)
dll_otf_restoration_cpp.get_average_angled_spectrum.argtypes = (np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2),
                                                                ct.c_size_t, ct.c_size_t,
                                                                np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                                                ct.c_size_t,
                                                                ct.c_double, ct.c_double)
dll_otf_restoration_cpp.get_average_angled_spectrum.restype = None
# Get average angled spectrum (CUDA)
dll_otf_restoration_cuda.get_average_angled_spectrum.argtypes = (np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2),
                                                                 ct.c_size_t, ct.c_size_t,
                                                                 np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                                                 ct.c_size_t,
                                                                 ct.c_double, ct.c_double)
dll_otf_restoration_cuda.get_average_angled_spectrum.restype = None
# Evaluates defocus parameter restoration functional (C++)
dll_otf_restoration_cpp.evaluate_fuctional.argtypes = (ct.c_double,
                                                       np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                                       np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                                       ct.c_size_t,
                                                       np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
                                                       ct.c_size_t,
                                                       ct.c_double, ct.c_double, ct.c_double)
dll_otf_restoration_cpp.evaluate_fuctional.restype = ct.c_double
# Evaluates defocus parameter restoration functional on grid (C++)
dll_otf_restoration_cpp.evaluate_fuctional_on_grid.argtypes = (np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                                               ct.c_size_t,
                                                               np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                                               np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                                               ct.c_size_t,
                                                               np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
                                                               ct.c_size_t,
                                                               ct.c_double, ct.c_double, ct.c_double)
dll_otf_restoration_cpp.evaluate_fuctional_on_grid.restype = None
# Evaluates defocus parameter restoration functional for RGB case (C++)
dll_otf_restoration_cpp.evaluate_fuctional_rgb.argtypes = (ct.c_double, ct.c_double,
                                                           np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                                           np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                                           ct.c_size_t,
                                                           np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
                                                           np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
                                                           ct.c_double,
                                                           np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                                           np.ctypeslib.ndpointer(dtype=np.float64, ndim=2), ct.c_size_t,
                                                           ct.c_double, ct.c_double)
dll_otf_restoration_cpp.evaluate_fuctional_rgb.restype = ct.c_double
# Evaluates defocus parameter restoration functional on grid for RGB case (C++)
dll_otf_restoration_cpp.evaluate_fuctional_on_grid_rgb.argtypes = (np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),
                                                                   np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                                                   ct.c_size_t,
                                                                   np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                                                   ct.c_size_t,
                                                                   np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                                                   np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                                                   ct.c_size_t,
                                                                   np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
                                                                   np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
                                                                   ct.c_double,
                                                                   np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),
                                                                   np.ctypeslib.ndpointer(dtype=np.float64, ndim=2), ct.c_size_t,
                                                                   ct.c_double, ct.c_double)
dll_otf_restoration_cpp.evaluate_fuctional_on_grid_rgb.restype = None


######################
# OTF Reconstruction #
######################


def average_angled_spectrum(image: np.ndarray, k=0.01, b=0.01) -> np.ndarray:
    """Computes average angled spectrum from image FFT.
    """

    # Half size of image as 1/2 of minimum from two sides
    half_size = min(image.shape[0], image.shape[1])//2

    # Copy observed image
    image_ = np.copy(image).astype(np.complex128)

    # Prepair averaged angled spectrum
    av_ang_spec = np.empty(half_size).astype(np.float64)

    # Make array C contiguous
    if not image_.flags['C_CONTIGUOUS']:
        image_ = np.ascontiguousarray(image_)
    if not av_ang_spec.flags['C_CONTIGUOUS']:
        av_ang_spec = np.ascontiguousarray(av_ang_spec)

    # Call DLL function (depending on image size)
    if image.shape[0] < 2048 and image.shape[1] < 2048:
        dll_otf_restoration_cpp.get_average_angled_spectrum(image_, *image_.shape,
                                                            av_ang_spec, ct.c_size_t(av_ang_spec.shape[0]),
                                                            ct.c_double(k), ct.c_double(b))
    else:
        dll_otf_restoration_cuda.get_average_angled_spectrum(image_, *image_.shape,
                                                             av_ang_spec, ct.c_size_t(av_ang_spec.shape[0]),
                                                             ct.c_double(k), ct.c_double(b))

    return av_ang_spec


def get_lower_envelope(x_points: np.ndarray, fuction_points: np.ndarray) -> np.ndarray:
    """Computes lower envelope of a given function.
    """

    # Both arrays should be 2D and have same shape
    assert(len(x_points.shape) == 1 and len(fuction_points.shape) == 1 and x_points.shape == fuction_points.shape)

    # Extend x array, by adding first its element to its start and last element to its end
    x_pad = np.empty(fuction_points.shape[0]+2)
    x_pad[1:-1], x_pad[:1], x_pad[-1:] = x_points, x_points[:1], x_points[-1:]

    # Extend function values array, by adding max value in array + 1 to its start and its end
    f_pad = np.empty(fuction_points.shape[0]+2)
    f_pad[1:-1], f_pad[::len(f_pad)-1] = fuction_points, np.max(fuction_points) + 1

    # Prepair 2D array
    data = np.column_stack((x_pad, f_pad))

    # Compute hull
    hull = ConvexHull(data)

    # Extract key points of envelope (in other points it is just a linear function)
    envelope_points = np.array([v-1 for v in hull.vertices if 0 < v <= fuction_points.shape[0]])

    # Return linear interpolation between known points
    return np.interp(x_points, [x_points[i] for i in envelope_points], [fuction_points[i] for i in envelope_points])


def normalize_by_envelope(function: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    """Normalizes function by its envelope (euclidean norm).
    """

    # Init normalized I(r)
    norm = np.empty(function.shape)

    # For every poit of I(r)
    for i in range(function.shape[0]):
        # First set initial value of minimum distance to envelope as distance to envelope point directly beneath
        norm[i] = function[i] - envelope[i]

        # For all envelope points
        for j in range(function.shape[0]):
            # Calculate distance to it (scales x axis to [0, 1])
            dist = math.sqrt((((i - j)/function.shape[0])**2 + (function[i] - envelope[j])**2))

            # Determine if new min was found
            if norm[i] > dist:
                norm[i] = dist

    return norm


def get_norm_zeros(function: np.ndarray, zeros_range_ratio: float) -> np.ndarray:
    """Extracts 'zeros' from given function.
    """

    # All local mins points
    mins_points = argrelextrema(function[:int(zeros_range_ratio * (function.shape[0]/math.pi))], np.less_equal)[0]
    # Leave only those that are close to 0
    mins_points = [i for i in mins_points if function[i] < 1e-3 and i > 1 and i < function.shape[0]-2]

    # Remove platos (adjacent indices corresponding to the same value)
    saved = set()
    zeros = []
    n = len(mins_points) - 1
    for i in range(len(mins_points)):
        # If this is not last index in a list and current and next indexes are adjacent and function value in them
        # is the same
        if i != n and mins_points[i + 1] - mins_points[i] == 1 and function[mins_points[i]] == function[mins_points[i + 1]]:
            # Save this index in a set
            saved.add(mins_points[i])
        else:
            # If set is not empty, insert its middle index
            if len(saved) != 0:
                # Do not forget to count current index
                saved.add(mins_points[i])
                saved = list(saved)
                zero = len(saved) // 2
                zeros.append(saved[zero])

                # Do not forget to reset set
                saved = set()
            else:
                # Insert current index
                zeros.append(mins_points[i])

    return np.array(zeros)


def evaluate_fuctional(a: float | np.ndarray, r: np.ndarray, norm: np.ndarray, norm_zeros: np.ndarray,
                       zeros_range_ratio: float, alpha=2.0, beta=1.0) -> float | np.ndarray:
    """Evaluates defocus parameter restoration functional for given defocus parameter (or defocus parameter grid).
    """

    # Make arrays C contiguous
    if not r.flags['C_CONTIGUOUS']:
        r = np.ascontiguousarray(r)
    if not norm.flags['C_CONTIGUOUS']:
        norm = np.ascontiguousarray(norm)
    if not norm_zeros.flags['C_CONTIGUOUS']:
        norm_zeros = np.ascontiguousarray(norm_zeros)

    # Check radius grid type
    if r.dtype != np.float64:
        r = r.astype(np.float64)
    # Check norm zeros type
    if norm_zeros.dtype != np.int32:
        norm_zeros = norm_zeros.astype(np.int32)

    # Check defocus parameter type
    if isinstance(a, float):
        # Call DLL function
        return dll_otf_restoration_cpp.evaluate_fuctional(ct.c_double(a), r, norm, norm.shape[0], norm_zeros, norm_zeros.shape[0],
                                                          ct.c_double(zeros_range_ratio), ct.c_double(alpha), ct.c_double(beta))
    if isinstance(a, np.ndarray):
        # Init result
        functional_values = np.copy(a)

        # Make arrays C contiguous
        if not functional_values.flags['C_CONTIGUOUS']:
            functional_values = np.ascontiguousarray(functional_values)

        # Call DLL function
        dll_otf_restoration_cpp.evaluate_fuctional_on_grid(functional_values, functional_values.shape[0], r, norm, norm.shape[0],
                                                           norm_zeros, norm_zeros.shape[0], ct.c_double(zeros_range_ratio),
                                                           ct.c_double(alpha), ct.c_double(beta))

        return functional_values


def evaluate_fuctional_rgb(a: float | np.ndarray, wlength0: float | np.ndarray, r: np.ndarray,
                           norm: list[np.ndarray, np.ndarray, np.ndarray], norm_zeros: list[np.ndarray, np.ndarray, np.ndarray],
                           zeros_range_ratio: float, rgb_ratios: np.ndarray, detector_funcs: np.ndarray,
                           alpha=2.0, beta=1.0) -> float | np.ndarray:
    """Evaluates defocus parameter restoration functional for given defocus parameter (or defocus parameter grid).
    """

    # Zeros count
    zeros_count = np.array([norm_zeros[0].shape[0], norm_zeros[1].shape[0], norm_zeros[2].shape[0]]).astype(np.int32)

    # Concatinated norm and zeros
    norm_concat = np.concatenate(norm)
    norm_zeros_concat = np.concatenate(norm_zeros)

    # Make arrays C contiguous
    if not r.flags['C_CONTIGUOUS']:
        r = np.ascontiguousarray(r)
    if not norm_concat.flags['C_CONTIGUOUS']:
        norm_concat = np.ascontiguousarray(norm_concat)
    if not norm_zeros_concat.flags['C_CONTIGUOUS']:
        norm_zeros_concat = np.ascontiguousarray(norm_zeros_concat)
    if not zeros_count.flags['C_CONTIGUOUS']:
        zeros_count = np.ascontiguousarray(zeros_count)
    if not rgb_ratios.flags['C_CONTIGUOUS']:
        rgb_ratios = np.ascontiguousarray(rgb_ratios)
    if not detector_funcs.flags['C_CONTIGUOUS']:
        detector_funcs = np.ascontiguousarray(detector_funcs)

    # Check radius grid type
    if r.dtype != np.float64:
        r = r.astype(np.float64)
    # Check norm zeros type
    if norm_zeros_concat.dtype != np.int32:
        norm_zeros_concat = norm_zeros_concat.astype(np.int32)

    # Check defocus parameter type
    if isinstance(a, float) and isinstance(wlength0, float):
        # Call DLL function
        return dll_otf_restoration_cpp.evaluate_fuctional_rgb(ct.c_double(a), ct.c_double(wlength0), r, norm_concat, norm[0].shape[0],
                                                              norm_zeros_concat, zeros_count, ct.c_double(zeros_range_ratio),
                                                              rgb_ratios, detector_funcs, detector_funcs.shape[1],
                                                              ct.c_double(alpha), ct.c_double(beta))
    if isinstance(a, np.ndarray) and isinstance(wlength0, np.ndarray):
        # Init result
        functional_values = np.empty((a.shape[0], wlength0.shape[0]))

        # Make arrays C contiguous
        if not functional_values.flags['C_CONTIGUOUS']:
            functional_values = np.ascontiguousarray(functional_values)
        if not a.flags['C_CONTIGUOUS']:
            a = np.ascontiguousarray(a)
        if not wlength0.flags['C_CONTIGUOUS']:
            wlength0 = np.ascontiguousarray(wlength0)

        # Call DLL function
        dll_otf_restoration_cpp.evaluate_fuctional_on_grid_rgb(functional_values, a, a.shape[0], wlength0, wlength0.shape[0],
                                                               r, norm_concat, norm[0].shape[0], norm_zeros_concat, zeros_count,
                                                               ct.c_double(zeros_range_ratio),
                                                               rgb_ratios, detector_funcs, detector_funcs.shape[1],
                                                               ct.c_double(alpha), ct.c_double(beta))

        return functional_values