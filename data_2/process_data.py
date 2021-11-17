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
root_path = 'images/'
region_paths = [root_path+f for f in os.listdir(root_path) if not f.startswith('.')]
image_dir_paths = []
for region in region_paths:
    for f in os.listdir(region):
        if not f.startswith('.'):
            image_dir_paths.append(region+'/'+f)
            
image_paths = []
for image_dir in image_dir_paths:
    files_in_dir = os.listdir(image_dir)
    for f in files_in_dir:
        if (f.endswith('.tiff') or f.endswith('.tif')) and not \
            (f.endswith('_rgb-nir.tiff') or f.endswith('_rgb-nir.tif')):
            image = image_dir + '/' + f
            image_paths.append(image)

# Apply sigmoidal color corrections to RGB/NIR-bands, this betters contrasts, but some atmospheric effects are more prominent in some images it seems
for img in image_paths:
    dat = rasterio.open(img)
    with rasterio.open(img) as dataset:
        image = dataset.read((4,3,2))
        image = image/255.0 # Normalize np-array RGB-values
        image = rc.sigmoidal(image,6,0.3)
        
    print(image[0,0:5,0])
    print(image[1,0:5,0])
    print(image[2,0:5,0])
    # print(image[3,0:5,0])
    raise Exception("Testing")
    dataset.close()
    dat.close()

    # Save new RGB-tiff to separate raster
    path = os.path.split(img)
    out_name = path[1].split('.')
    out_name = out_name[0] + '_rgb-nir.' + out_name[1]
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
        if f.endswith('_rgb-nir.tiff') or f.endswith('_rgb-nir.tif'):
            image = image_dir + '/' + f
            image_paths_rgb.append(image)

# Tiling
for img in image_paths_rgb:
    root_path = os.path.split(img)[0]
    if not os.path.exists(root_path+'/tiled_images'): # Check if already tiled
        rpt.tile_and_save(img,size=(64,64),data='image')

# Convert images to JPEG/PNG:

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
root_path = 'masks/'
mask_paths = [root_path+f for f in os.listdir(root_path) if not f.startswith('.')]

# Reproject rasters to EPSG:4326 to match satellite rasters
for mask in mask_paths:
    name = [f for f in os.listdir(mask) if not f.startswith('.')][0]
    name = name.split('.')[0]
    mask_path = mask+'/'+name
    if not os.path.exists(mask_path+'_proj.tif'):
        rpt.reproject_raster(mask_path+'.tif',crs='EPSG:4326')

# Crop reprojected mask rasters
reprojected_masks = {
    'norway': ('masks/norway/esri_norway_proj.tif', (10.194969,63.154200,10.371094,63.233937)),
    'egypt': ('masks/egypt/esri_egypt_proj.tif', (32.395248,30.137705,32.580986,30.311542)),
    'sudan': ('masks/sudan/esri_sudan_proj.tif', (31.301079,21.753441,31.451454,21.925848))
}
# Crop
rpt.crop_GTiff(reprojected_masks['norway'][0],bbox=reprojected_masks['norway'][1])
rpt.crop_GTiff(reprojected_masks['egypt'][0],bbox=reprojected_masks['egypt'][1])
rpt.crop_GTiff(reprojected_masks['sudan'][0],bbox=reprojected_masks['sudan'][1])

# Match resolution of mask to satelitte images:
cropped_masks = {
    'norway': ('masks/norway/esri_norway_proj_crop.tif'),
    'egypt': ('masks/egypt/esri_egypt_proj_crop.tif'),
    'sudan': ('masks/sudan/esri_sudan_proj_crop.tif')
}

image = 'images/norway/2019-05-02:2019-06-02/2019-05-02:2019-06-02.tiff'
rpt.match_resolution(image,cropped_masks['norway'])
image = 'images/egypt/2019-01-01:2019-01-31/2019-01-01:2019-01-31.tiff'
rpt.match_resolution(image,cropped_masks['egypt'])
image = 'images/sudan/2019-11-01:2019-12-01/2019-11-01:2019-12-01.tiff'
rpt.match_resolution(image,cropped_masks['sudan'])

# Tile matched mask raster:
processed_masks = {
    'norway': ('masks/norway/esri_norway_projcrop_matched.tif'),
    'egypt': ('masks/egypt/esri_egypt_projcrop_matched.tif'),
    'sudan': ('masks/sudan/esri_sudan_projcrop_matched.tif')
}

rpt.tile_and_save(processed_masks['norway'],size=(64,64),data='mask')
rpt.tile_and_save(processed_masks['egypt'],size=(64,64),data='mask')
rpt.tile_and_save(processed_masks['sudan'],size=(64,64),data='mask')
        
# Convert masks to JPEG/PNG:

# Full raster
rpt.Gtiff2rgb(processed_masks['norway'],outFormat='jpeg',bands=1)
rpt.Gtiff2rgb(processed_masks['egypt'],outFormat='jpeg',bands=1)
rpt.Gtiff2rgb(processed_masks['sudan'],outFormat='jpeg',bands=1)

# Tiles
mask_tile_paths = {
    'norway': 'masks/norway/tiled_masks/',
    'egypt': 'masks/egypt/tiled_masks/',
    'sudan': 'masks/sudan/tiled_masks/'
}

mask_tiles_nor = [mask_tile_paths['norway'] + '/' + f for f in os.listdir(mask_tile_paths['norway']) if f.endswith('.tif')]
for tile in mask_tiles_nor:
    rpt.Gtiff2rgb(tile,outFormat='jpeg',bands=1)
    
mask_tiles_egypt = [mask_tile_paths['egypt'] + '/' + f for f in os.listdir(mask_tile_paths['egypt']) if f.endswith('.tif')]
for tile in mask_tiles_egypt:
    rpt.Gtiff2rgb(tile,outFormat='jpeg',bands=1)
    
mask_tiles_sud = [mask_tile_paths['sudan'] + '/' + f for f in os.listdir(mask_tile_paths['sudan']) if f.endswith('.tif')]
for tile in mask_tiles_sud:
    rpt.Gtiff2rgb(tile,outFormat='jpeg',bands=1)
