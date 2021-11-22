import rasterio
from rasterio.plot import show

img = rasterio.open('data/train/images/6.tif').read()
mask = rasterio.open('data/train/masks/6.tif').read()

show(img)
show(mask)