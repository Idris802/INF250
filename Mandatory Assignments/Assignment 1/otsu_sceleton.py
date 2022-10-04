# -*- coding: utf-8 -*-

__author__ = "Mohammed Idris Omar"
__email__ = "mohammed.idris.omar@nmbu.no"


import numpy as np
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


def threshold(image, th=None):
    """Returns a binarised version of given image, thresholded at given value.
    Binarises the image using a global threshold `th`. Uses Otsu's method
    to find optimal thrshold value if the threshold variable is None. The
    returned image will be in the form of an 8-bit unsigned integer array
    with 255 as white and 0 as black.
    Parameters:
    -----------
    image : np.ndarray
        Image to binarise. If this image is a colour image then the last
        dimension will be the colour value (as RGB values).
    th : numeric
        Threshold value. Uses Otsu's method if this variable is None.
    Returns:
    --------
    binarised : np.ndarray(dtype=np.uint8)
        Image where all pixel values are either 0 or 255.
    """
    # Setup
    shape = np.shape(image)

    if len(shape) == 3:
        image = image.mean(axis=2)
    elif len(shape) > 3:
        raise ValueError('Must be at 2D image')

    if th is None:
        th = otsu(image)

    return (image > th) * 255


def histogram(image):
    shape = np.shape(image)

    if len(shape) != 2:
        raise ValueError('Must be a 2D image')

    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))

    return histogram


def otsu(image):
    """Finds the optimal thresholdvalue of given image using Otsu's method.
    """
    shape = np.shape(image)
    height, width = shape[0], shape[1]
    hist = histogram(image)

    sum_all = 0
    for t in range(256):
        sum_all += t * hist[t]

    sum_back, weight_b, weight_f, var_max, th = 0, 0, 0, 0, 0
    total = height * width

    for t in range(256):

        weight_b += hist[t]
        if weight_b == 0:
            continue

        weight_f = total - weight_b
        if weight_f == 0:
            break

        sum_back += t * hist[t]
        mean_b = sum_back / weight_b
        mean_f = (sum_all - sum_back) / weight_f

        var_b = weight_b * weight_f * (mean_b - mean_f) ** 2

        if var_b > var_max:
            var_max = var_b
            th = t

    return th


def create_plots(image, th):
    image_gray = rgb2gray(image)
    hist = histogram(image_gray)
    binary = threshold(image, th)

    fig, axes = plt.subplots(ncols=3, figsize=(10, 3))
    axes[0].imshow(image_gray, cmap='gray')
    axes[0].set_title("Original")

    axes[1].plot(hist)
    axes[1].set_title("Histogram")

    axes[2].imshow(binary, cmap='gray')
    axes[2].set_title("Thresholded")

    plt.savefig("Orig_Hist_Thres_applied", format="png")
    plt.show()


if '__main__' == __name__:
    image = io.imread('gingerbreads.jpg')
    grayscale = rgb2gray(image)
    th = otsu(grayscale)
    create_plots(image, th)
