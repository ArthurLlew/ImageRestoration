###########
# IMPORTS #
###########


# File operations
import os
# Math
import math
# Handy arrays
import numpy as np
# Image loading/saving
import matplotlib.image as MPLI


#########
# Utils #
#########


#================
#== Validation ===
#================


def check_shape_is_2d(shape: tuple) -> None:
    """Validates, that the given numpy shape is 2D.
    """
    
    if len(shape) != 2:
        raise Exception('Shape is not 2D')


def check_shape_is_simmentrical(shape: tuple) -> None:
    """Validates, that the given numpy shape is 2D and simmentrical.
    """
    
    # Only 2D arrays are accepted
    check_shape_is_2d(shape)
    
    if shape[0] != shape[1]:
        raise Exception('Shape is not simmentrical')


def check_array_is_complex(array: np.ndarray) -> None:
    """Validates, that the given numpy array is complex.
    """
    
    if array.dtype != np.dtype('complex'):
        raise Exception('Array is not complex')


def check_array_is_not_complex(array: np.ndarray) -> None:
    """Validates, that the given numpy array is not complex.
    """
    
    if array.dtype == np.dtype('complex'):
        raise Exception('Array is suddenly complex')


def check_shape_is_rgb(shape: tuple) -> None:
    """Validates, that the given numpy shape is 2D.
    """
    
    # 3D array + last dimension equals 3 (number of channels)
    if not(len(shape) == 3 and shape[2] == 3):
        raise Exception('Shape is not in RGB format')


#====================================
#== Matrix indexes and coordinates ===
#====================================


def create_matrix_grid_indexes(shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Creates two grid arrays of indexes.
    """
    
    return np.ogrid[:shape[0],:shape[1]]


def create_matrix_grid_coordinates(shape: tuple, symmetry=False) -> tuple[np.ndarray, np.ndarray]:
    """Creates two grid arrays of coordinates (mapped to [-pi, pi]).
    """

    # Matrix indexes
    I,J = create_matrix_grid_indexes(shape)

    # Return grid, mapped to [-pi, pi]
    if (not symmetry) or (shape[0]%2 == 1):
        I = (1 - I/(shape[0]//2)) * math.pi
    else:
        # Put zero between middle pixels if sizes are even numbers
        I = np.empty((shape[0], 1))
        ln = np.linspace(math.pi / shape[0], math.pi, shape[0]//2)
        I[shape[0]//2:, 0] = -ln
        I[:shape[0]//2, 0] = np.flip(ln)
    if (not symmetry) or (shape[1]%2 == 1):
        J = (J/(shape[1]//2) - 1) * math.pi
    else:
        # Put zero between middle pixels if sizes are even numbers
        J = np.empty((1, shape[1]))
        ln = np.linspace(math.pi / shape[1], math.pi, shape[1]//2)
        J[0, shape[1]//2:] = ln
        J[0, :shape[1]//2] = -np.flip(ln)

    # I corresponds to row (Y), and J - to column (X)
    return I, J


#===================
#== Image loading ===
#===================


def image_crop(image: np.ndarray, cropping=None) -> np.ndarray:
    """Crops image using provided slices. If cropping is None, the biggest square in the image center is chosen.
    """
    
    if cropping is None:
        # Use square in image center
        
        # Min/max sizes and their axis
        min_size, max_size = min(image.shape[0:2]), max(image.shape[0:2])
        min_ax, max_ax = np.argmin(image.shape[0:2]), np.argmax(image.shape[0:2])
        
        # Shift from start for larger axis
        shift = (max_size - min_size) // 2
        
        # Slices
        s1 = slice(0, min_size)
        s2 = slice(shift, min_size + shift)
        
        # Choose slice order
        if min_ax < max_ax:
            image = image[(s1, s2)]
        else:
            image = image[(s2, s1)]
    else:
        # Crop if cropping was provided
        image = image[(cropping[0], cropping[1])]
    
    return image


def image_load(path: str, cropping=None, mode='grayscale') -> np.ndarray:
    """Loads image, crops it and normalizes to [0, 1].
    """
    
    # Load image as rgb
    image = np.array(MPLI.imread(path), dtype=np.float64)
    # Remove Alpha channel
    image = image[:, :, :3]
    
    if mode=='grayscale':
        if len(image.shape) > 2:
            image = rgb2grayscale(image)
    
    # Normalize
    if image.max() > 1:
        image = image / 255

    # Return cropped image
    return image_crop(image, cropping=cropping)


def image_save(image: np.ndarray, path: str, mode='grayscale') -> np.ndarray:
    """Saves image.
    """

    # Make sure directory exists
    path_split = path.rsplit('/', 1)
    if len(path_split) > 1:
        dir = path_split[0]
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    if mode=='grayscale':
        MPLI.imsave(path, image, cmap='gray')
    
    if mode=='rgb':
        MPLI.imsave(path, image)


#===========================
#== Operations over image ===
#===========================


def image_normalize(image: np.ndarray, snap_min_max=False) -> np.ndarray:
    """Normalizes image to [0, 1].
    """

    image_ = np.copy(image)

    # Snap min to zero
    if (image.min() < 0 or snap_min_max):
        image_ = image_ - image_.min()
    # Divide by new max
    if (image_.max() > 1 or snap_min_max):
        image_ = image_ / image_.max()

    return image_


def image_crop_corners(image: np.ndarray, ratio=0.04) -> np.ndarray:
    """Crops image corners with provided ratio.
    """
    
    shape0 = int(image.shape[0]*ratio)
    shape1 = int(image.shape[1]*ratio)
    return image[shape0:-shape0, shape1:-shape1]


def rgb_mul_mono(image: np.ndarray, window_func: np.ndarray) -> np.ndarray:
    """Multiplies each channel of RGB image by given window function.
    """
    
    image[:,:,0] *= window_func
    image[:,:,1] *= window_func
    image[:,:,2] *= window_func
    return image_normalize(image)


def image_max_via_histogram(image: np.ndarray) -> float:
    """Calculates maximum of monochrome image via histogram.
    """

    if len(image.shape) != 2:
        raise Exception('Image is not monochrome')

    # Histogram
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))
    # Size of image point to produce max
    point_size = 0.0004 * image.shape[0] * image.shape[1]

    max_val = 0
    for i in np.flip(np.arange(256)):
        if histogram[i] >= point_size:
            max_val = i
            break

    return bin_edges[max_val+1]


def image_rgb_max_via_histogram(image: np.ndarray) -> np.ndarray:
    """Calculates maximums of image channels via histogram.
    """

    if len(image.shape) != 3:
        raise Exception('Image is not rgb')

    # Max per channel
    return np.array([image_max_via_histogram(image[:, :, 0]),
                     image_max_via_histogram(image[:, :, 1]),
                     image_max_via_histogram(image[:, :, 2])])


def get_squared_complex_module(image_fft: np.ndarray) -> np.ndarray:
    """Calculates squared complex module element wise.
    """
    
    return (image_fft * np.conjugate(image_fft)).real


def get_spectrum(image_fft: np.ndarray) -> np.ndarray:
    """Calculates energy spectrum of the image FFT.
    """
    
    # Get squared complex module
    spectrum = get_squared_complex_module(image_fft)

    # Return log scaled values
    return np.log(1 + spectrum)


def rgb2grayscale(image: np.ndarray) -> np.ndarray:
    """Grayscales image.
    """

    return image_normalize(np.dot(image, [0.299, 0.587, 0.144]))


def image_autocontrast(image: np.ndarray, threshold=0.005) -> np.ndarray:
    """Performes autocontrast: determines new image min and max via histogram, clamps values by them and normalizes result.
    """

    # Copy image
    image = np.copy(image)

    # Convert ratio to pixels
    threshold_px = threshold * image.shape[0] * image.shape[1]

    # Helper function
    def clamp_by_histogram(image: np.ndarray) -> np.ndarray:
        # Histogram
        histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))

        # New image minimum from histogram
        i = 0
        while (histogram[i] < threshold_px) and (i < 255):
            i += 1
        new_min = bin_edges[i]

        # New image maximum from histogram
        i = 255
        while (histogram[i] < threshold_px) and (i > 1):
            i -= 1
        new_max = bin_edges[i]

        # Clamp image
        return((image < new_min) * new_min
               + (image > new_max) * new_max
               + image * ((image >= new_min) * (image <= new_max)))

    # Depending on image
    if (len(image.shape) == 2):
        # Grayscale
        image = clamp_by_histogram(image)
    elif (len(image.shape) == 3):
        # For each channel
        for ch in range(3):
            image[:,:,ch] = clamp_by_histogram(image[:,:,ch])

    # Normalize image with values snapping
    image = image_normalize(image, snap_min_max=True)

    return image


#==========
#== Math ===
#==========


def L2_norm(array: np.ndarray) -> float:
    """Computes L2 norm of provided array.
    """

    # Compute integral step
    h_ij = 1
    for size in array.shape:
        h_ij *= (math.pi / size)

    # Sqrt from intergral of squared function (L2 norm formula)
    return math.sqrt(np.sum((array)**2 * h_ij))


#====================
#== Gauss function ===
#====================


def create_gauss_1d(wlgrid: np.ndarray, wlength0: float, sigma: float) -> np.ndarray:
    """Evaluates 1D gauss function over given grid.
    """
    
    # Formula
    return np.exp(-(wlgrid - wlength0)**2 / sigma**2)


def create_gauss_1d_3bunched(wlgrid: np.ndarray, wlength0s: tuple[float, float, float], sigmas: tuple[float, float, float]) -> np.ndarray:
    """Evaluates three 1D gauss functions with given parameters in given grids and returns result as a packed tuple.
    """
    
    # Fill values list
    gauss_functions = [wlgrid]
    for i in range(3):
        gauss_functions.append(create_gauss_1d(wlgrid, wlength0s[i], sigmas[i]))
    
    return np.array(gauss_functions)


def create_super_gauss(shape: tuple, sigma: float, p: float, mode='round') -> np.ndarray:
    """Creates 2D super-gauss function with given params.
    """
    
    # Create grid
    X,Y = create_matrix_grid_coordinates(shape, symmetry=True)
    
    # Scale sigma
    sigma = sigma * math.pi
    p = p * math.pi
    
    if mode == 'round':
        # Formula
        return np.exp(-(((X/sigma)**2 + (Y/sigma)**2)**p))
    elif mode == 'square':
        # Formula
        return np.exp(-(((X/sigma)**2)**p + ((Y/sigma)**2)**p))
    else:
        raise Exception('Unrecognized mode')


#======================
#== Precedural image ===
#======================


def create_image(shape: tuple, n=12, mode='', sharp=True) -> np.ndarray:
    """Creates custom spiral image.
    """
    
    # Generate rays
    X,Y = create_matrix_grid_coordinates(shape, symmetry=True)
    image = np.sin(np.arctan(Y/X) * n)
    if sharp:
        image = (image >= 0).astype(np.float64)
    
    # If image should be RGB
    if mode == 'rgb':
        # Copy image
        pre_image = image
        
        # Fill channels
        image = np.empty((pre_image.shape[0], pre_image.shape[1], 3))
        image[:,:,0] = pre_image
        image[:,:,1] = pre_image
        image[:,:,2] = pre_image
    
    # Return normalized image
    return image_normalize(image)