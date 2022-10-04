from skimage import io
import matplotlib.pyplot as plt
from skimage.filters import gaussian, threshold_otsu
from skimage.color import rgb2gray

fountain_image = "fountain.jpg"
read_image = io.imread(fountain_image)

# Applying gaussian filter to the image with sigma value of 5
gaussian_filtered = gaussian(read_image, sigma=5, channel_axis=False)

plt.figure()
plt.imshow(read_image)
plt.figure()
plt.imshow(gaussian_filtered)
plt.show()

# Applying an Otsu thresholded image
grayscale = rgb2gray(read_image)
thresh = threshold_otsu(grayscale)
binary = grayscale > thresh

fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 3, 1)
ax[1] = plt.subplot(1, 3, 2)
ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

ax[0].imshow(read_image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].hist(read_image.ravel(), bins=256)
ax[1].set_title('Histogram')
ax[1].axvline(thresh, color='r')

ax[2].imshow(binary, cmap=plt.cm.gray)
ax[2].set_title('Thresholded')
ax[2].axis('off')

plt.show()