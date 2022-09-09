from skimage.viewer import ImageViewer
from skimage import io
import matplotlib.pylab as plt

# reading image
filename = 'fall.tiff'
lift = io.imread(filename)

# display of image axis
print(lift.shape)

# extracting one colour and finding min and max values
lift_red = lift[:, :, 0]
print(max(lift_red.flatten()))
print(min(lift_red.flatten()))

plt.imshow(lift_red, vmin=0, vmax=255)
plt.show()

# another viewer
viewer = ImageViewer(lift)
viewer.show()