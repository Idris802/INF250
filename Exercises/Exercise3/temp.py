# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt 
from skimage import io
from skimage.filters import threshold_otsu

read_image = io.imread("airfield.tif")
plt.figure()
plt.imshow(read_image, cmap="gray")

def histogram(image):
    shape = np.shape(image)
    histogram = np.zeros(256)
    for i in range(shape[0]):
        for j in range(shape[1]):
            pixval = int(image[i,j])
            histogram[pixval] += 1
    
    return histogram

hist = histogram(read_image)
plt.figure()
plt.plot(hist)


# 4) Carry out a histogram equalisation. Has the image changed ?

# First take the cumalitive histogram of the histogram.
def cumulative_hist(histogram):
    cumhist = np.zeros(256)
    cumhist[0] = histogram[0]
    for i in range(255):
        cumhist[i+1] = cumhist[i] + histogram[i+1]
    return cumhist

c_h = cumulative_hist(hist)
plt.figure()
plt.plot(c_h)

# Now carrying out the equalization 
def equalise_hist(cumhist, image):
    shape = np.shape(image)
    K_bins = 256
    M_height = shape[0] 
    N_widht = shape[1]
    M_N = M_height * N_widht
    for i in range(shape[0]):
        for j in range(shape[1]):
            a = int(image[i,j])
            b = cumhist[a]*(K_bins-1)/M_N
            image[i, j] = b
    return cumhist, image

equalise_histogram, image_equalized = equalise_hist(c_h, read_image)
plt.figure()
plt.imshow(image_equalized, cmap='gray') 

# 5) plot the histogram of the modified image
plt.figure()
plt.plot(equalise_histogram)

# 6) Carry out an Otsu threshold on the image, you can programme it on you own or use the function in Scikit-image.
threshold_value_otsu = threshold_otsu(read_image)
binary = read_image > threshold_value_otsu
plt.figure()
plt.imshow(binary, cmap='gray')
