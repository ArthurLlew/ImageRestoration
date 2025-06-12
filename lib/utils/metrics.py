###########
# IMPORTS #
###########


# Handy arrays
import numpy as np
# SSIM and CW SSIM metrics
import ssim
# PIL image type
from PIL import Image
# Custom modules
import lib.utils.utils as utils


#################
# Metrics Utils #
#################


def numpy_to_pill(image: np.ndarray) -> Image:
    """Converts numpy array to PIL image (image should be from 0.0 to 1.0).
    """
    
    if np.min(image) != 0. or image.max() != 1.:
        raise Exception('Image is not from 0.0 to 1.0')
    
    return Image.fromarray(np.uint8(image*255))


def compare_images_mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compares images using MSE metrics.
    """
    
    return np.mean((image1 - image2)**2)


def compare_images_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compares images using PSNR metrics.
    """
    
    return 20 * np.log10(1 / np.sqrt(compare_images_mse(image1, image2)))


def compare_images_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compares images using SSIM metrics.
    """
    
    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel_1d = ssim.utils.get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)
    
    return ssim.SSIM(numpy_to_pill(image1), gaussian_kernel_1d).ssim_value(numpy_to_pill(image2))


def compare_images_cw_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compares images using CW SSIM metrics.
    """
    
    return ssim.SSIM(numpy_to_pill(image1)).cw_ssim_value(numpy_to_pill(image2))


def image_dx_grad_norm(image: np.ndarray) -> float:
    """Computes l2 norm of the image gradient by dx (df/dx).
    """

    # df/dx
    grad = np.array(np.gradient(image, axis=0))

    # Grayscale
    if (len(image.shape) == 2):
        # Grad L2 norm
        return utils.L2_norm(grad)
    # RGB
    elif (len(image.shape) == 3):
        # Sum of each channel grad L2 norm
        grad_norm = 0.0
        for ch in range(3):
            grad_norm += utils.L2_norm(grad[:,:,ch])
        return grad_norm
    # Unsupported operation
    else:
        raise ValueError('Image must be either grayscale or RGB')


def image_dy_grad_norm(image: np.ndarray) -> float:
    """Computes l2 norm of the image gradient by dy (df/dy).
    """

    # df/dy
    grad = np.array(np.gradient(image, axis=1))

    # Grayscale
    if (len(image.shape) == 2):
        # Grad L2 norm
        return utils.L2_norm(grad)
    # RGB
    elif (len(image.shape) == 3):
        # Sum of each channel grad L2 norm
        grad_norm = 0.0
        for ch in range(3):
            grad_norm += utils.L2_norm(grad[:,:,ch])
        return grad_norm
    # Unsupported operation
    else:
        raise ValueError('Image must be either grayscale or RGB')


def image_dz_grad_norm(image: np.ndarray) -> float:
    """Computes l2 norm of the image gradient by dz (df/dz).
    """

    # df/dz
    grad = np.array(np.gradient(image, axis=2))

    # Grayscale
    if (len(image.shape) == 2):
        # Grad L2 norm
        return utils.L2_norm(grad)
    # RGB
    elif (len(image.shape) == 3):
        # Sum of each channel grad L2 norm
        grad_norm = 0.0
        for ch in range(3):
            grad_norm += utils.L2_norm(grad[:,:,ch])
        return grad_norm
    # Unsupported operation
    else:
        raise ValueError('Image must be either grayscale or RGB')