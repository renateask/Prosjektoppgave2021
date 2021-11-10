import os
import rio_color.operations as rc
import raster_preprocess_tools as rpt
import rasterio
import warnings
warnings.filterwarnings('ignore')

"""
In this script:

- Improve image contrast
- Tile images
- Convert images and tiles to 3-band JPEG/PNG
- Reproject full mask to ESPG4326:WGS84
- Crop mask to satellite image coordinates
- Match pixel resolution of mask to satellite image for alignment
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
    dataset.close()
    dat.close()

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
        dst.close()

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
for img in image_paths_rgb:
    root_path = os.path.split(img)[0]
    if not os.path.exists(root_path+'/tiled_images'): # Check if already tiled
        rpt.tile_and_save(img,size=(64,64),data='image')

# Convert images to JPEG:

# Full raster
for image in image_paths_rgb:
    rpt.Gtiff2rgb(image,outFormat='jpeg',bands=3)

# Tiles
for image_dir in image_dir_paths:
    tile_path = image_dir + '/tiled_images/'
    tiles = [tile_path+f for f in os.listdir(tile_path) if f.endswith('.tif')]
    for tile in tiles:
        rpt.Gtiff2rgb(tile,outFormat='jpeg',bands=3)


# Define mask locations
mask_path = 'masks/esriNOR.tif'

# Reproject raster to EPSG:4326 to match satellite rasters
if not os.path.exists('masks/esriNOR_proj.tif'):
    rpt.reproject_raster(mask_path,crs='EPSG:4326')

# Crop reprojected mask raster
mask_reprojected_path = 'masks/esriNOR_proj.tif'
# Define EPSG4326:WGS84 coordinate bounding box equal to that of satellite raster
box = (10.194969,63.154200,10.371094,63.233937)
# Crop
print("I'm here!")
rpt.crop_GTiff(mask_reprojected_path,bbox=box)
print("After")

# Match resolution of mask to satelitte image:
mask_clip_reproj_path = 'masks/esriNOR_proj_crop.tif'
image = image_paths[0]
rpt.match_resolution(image,mask_clip_reproj_path)

# Tile matched mask raster:
processed_mask_path = 'masks/esriNOR_projcrop_matched.tif'
rpt.tile_and_save(processed_mask_path,size=(64,64),data='mask')
        
# Convert masks to JPEG:

# Full raster
rpt.Gtiff2rgb(processed_mask_path,outFormat='jpeg',bands=1)

# Tiles
mask_tile_path = 'masks/tiled_masks/'
mask_tiles = [mask_tile_path + f for f in os.listdir(mask_tile_path) if f.endswith('.tif')]
for tile in mask_tiles:
    rpt.Gtiff2rgb(tile,outFormat='jpeg',bands=1)




