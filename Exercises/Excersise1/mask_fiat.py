import numpy as np
from skimage.viewer import ImageViewer
from skimage import io
import matplotlib.pyplot as plt

infile = 'fiat.jpg'
fiat = io.imread(infile)

lift_red = fiat[:, :, 0]
print(max(lift_red.flatten()))
print(min(lift_red.flatten()))
print(lift_red.mean())
print(f'The shape of the full color image is {fiat.shape}')

fiat[5, 3] = 0


nrows, ncols = fiat.shape[0], fiat.shape[1]
row, col = np.ogrid[:nrows, :ncols]
cnt_row, cnt_col = nrows / 2, ncols / 2
outer_disk_mask = ((row*1.5 - cnt_row*1.5)**2 + (col*1.5 - cnt_col*1.5)**2 > (nrows / 2)**2)
fiat[outer_disk_mask] = 0


viewer = ImageViewer(fiat)
viewer.show()