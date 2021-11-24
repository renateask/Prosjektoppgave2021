import rasterio
import os
import numpy as np
from matplotlib import pyplot as plt
from rasterio.plot import show

train_tiles = ['train/images/'+f for f in os.listdir('train/images') if f.endswith('.tif')]

for tile in train_tiles:
    img = rasterio.open(tile).read()
    img = np.rollaxis(img,0,3)
    mask = rasterio.open(tile.replace('images','masks')).read()
    mask = np.rollaxis(mask,0,3)
    fig, ax = plt.subplots(2)
    ax[0].imshow(img)
    ax[1].imshow(mask)
    plt.show()

