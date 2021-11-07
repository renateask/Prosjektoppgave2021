import os
import rio_color.operations as rc
import raster_preprocess_tools as rpt
from rasterio.plot import show
import rasterio

"""
In this script:

- Improve image brightness
- Tile images
- Convert images and tiles to 3-band JPEG/PNG
- Rotate mask
- Tile mask
- Convert mask tiles to JPEG/PNG
"""

# Define image locations
image_paths = []
image_dir_paths = ['images/' + f for f in os.listdir('images/') if not f.startswith('.')]
for image_dir in image_dir_paths:
    files_in_dir = os.listdir(image_dir)
    for f in files_in_dir:
        if (f.endswith('.tiff') or f.endswith('.tif')) and not \
            (f.endswith('_rgb.tiff') or f.endswith('_rgb.tif')):
            image = image_dir + '/' + f
            image_paths.append(image)

# Apply sigmoidal color corrections to RGB-bands, this betters contrasts, but some atmospheric effects are more prominent in some images it seems
for img in image_paths:
    dat = rasterio.open(img)
    with rasterio.open(img) as dataset:
        image = dataset.read((4,3,2))
        image = image/255.0 # Normalize np-array RGB-values
        image = rc.sigmoidal(image,6,0.3)
        image = image*255.0

    # Save new RGB-tiff to separate raster
    path = os.path.split(img)
    out_name = path[1].split('.')
    out_name = out_name[0] + '_rgb.' + out_name[1]
    if not os.path.exists(path[0]+'/'+out_name):
        with rasterio.open(
            path[0]+'/'+out_name,'w',
            driver='GTiff',
            height=image.shape[1],
            width=image.shape[2],
            count=image.shape[0],
            dtype=image.dtype,
            crs=dat.crs,
            transform=dat.transform,
            ) as dst:
            dst.write(image)

# Tile RGB Tiffs to 64x64 chuncks:

# New image paths
image_paths_rgb = []
for image_dir in image_dir_paths:
    files_in_dir = os.listdir(image_dir)
    for f in files_in_dir:
        if f.endswith('_rgb.tiff') or f.endswith('_rgb.tif'):
            image = image_dir + '/' + f
            image_paths_rgb.append(image)

# Tiling
for img in image_paths:
    root_path = os.path.split(img)[0]
    if not os.path.exists(root_path+'/tiled_images'): # Check if already tiled
        rpt.tile_and_save(img,size=(64,64),data='image')

# Convert images to JPEG:

# Full raster
for image in image_paths_rgb:
    rpt.Gtiff2rgb(image,outFormat='jpeg')

# Tiles
for image_dir in image_dir_paths:
    tile_path = image_dir + '/tiled_images/'
    tiles = [tile_path+f for f in os.listdir(tile_path) if not f.startswith('.')]
    for tile in tiles:
        rpt.Gtiff2rgb(tile,outFormat='jpeg')







# Define mask locations
mask_dir_paths = ['masks/' + f for f in os.listdir('masks/') if not f.startswith('.')]






