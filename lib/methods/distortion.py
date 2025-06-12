###########
# IMPORTS #
###########


# Handy arrays
import numpy as np
# Images
import cv2
# FFTs
import scipy.fft as fft
# Custom modules
import lib.utils.utils as utils
import lib.utils.otf as OTF


######################
# Distortion Methods #
######################


def image_distortion(image: np.ndarray, a: float, additive_noise=None, noise_ratio=0.03) -> np.ndarray:
    """Distorts image using OTF and provided additive noise.
    """

    otf = OTF.create_otf(image.shape, a)
    
    # Real part of IFFT from multiplication of image FFT and OTF multiplication
    image_blurred = (fft.ifft2(fft.fft2(image) * otf)).real
    
    # Normalize
    utils.image_normalize(image_blurred)
    
    # Add additive noise if it is required
    if additive_noise is not None:
        image_blurred = cv2.addWeighted(image_blurred, 1 - noise_ratio, noise_ratio*additive_noise, noise_ratio, 0)
    
    return image_blurred


def image_distortion_rgb(image: np.ndarray, otf_batch: np.ndarray, additive_noise=None, noise_ratio=0.02) -> np.ndarray:
    """Distorts RGB image using OTF batch and provided additive noise.
    """
    
    # Constructing initial blurred image
    image_blurred = np.empty(image.shape)
    for channel_i in range(3):
        # Multiplication of image channels FFTs by OTF patches
        channels = np.empty(image.shape, dtype=np.complex128)
        for channel_j in range(3):
            channels[:, :, channel_j] = fft.fft2(image[:, :, channel_j]) * otf_batch[channel_i, channel_j, :, :]
        
        # IFFT real part of mean
        image_blurred[:, :, channel_i] = fft.ifft2(np.sum(channels, axis=2)).real
    
    # Normalize image
    image_blurred = utils.image_normalize(image_blurred)
    
    # Add additive noise if it is required
    if additive_noise is not None:
        image_blurred = cv2.addWeighted(image_blurred, 1 - noise_ratio, additive_noise, noise_ratio, 0)
        
        # Normalize image
        image_blurred = utils.image_normalize(image_blurred)
    
    return image_blurred