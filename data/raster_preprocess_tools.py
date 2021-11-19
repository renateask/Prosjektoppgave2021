import os
import subprocess
import json
from itertools import product

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import windows

def get_tiles(ds, width=64, height=64):
    """
    Provide tile window and transform.

    Used by internal methods only, even though access-restriction is not provided.
    """
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in  offsets:
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform

def tile_write(image,output_path,size=(64,64)):
    """
    Tile large satellite image and save to specified output location.

    ----- Parameters: -----
    image (full_size .tif)
    output_path
    size (tuple) :: (height,width) of desired tiles
    """
    output_filename = 'tile_{}-{}.tif'
    meta = image.meta.copy()
    
    for window, transform in get_tiles(image,size[0],size[1]):
        print(window)
        meta['transform'] = transform
        if window.width == size[1] and window.height == size[0]:
            meta['width'],meta['height'] = window.width,window.height
            outpath = os.path.join(output_path,output_filename.format(int(window.col_off), int(window.row_off)))
            with rasterio.open(outpath, 'w', **meta) as outds:
                outds.write(image.read(window=window))

def tile_and_save(image_path,size=(64,64),data='image'):
    """
    Read Geotiff image from sentinelhub, append image to list.

    ----- Parameters: -----
    path (str) :: location of image folders
    bands (tuple) :: which multispectral image bands to read (default: RGB)
    tile (bool) :: if True --> tile image and save image tiles
    """
    img = rasterio.open(image_path)
    region = os.path.split(image_path)[1].split('_')[0]
    if data == 'image':
        out_path = os.path.split(image_path)[0]+'/tiled_images/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    elif data == 'mask':
        out_path = os.path.split(image_path)[0]+'/tiled_masks_'+region+'/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    tile_write(img,output_path=out_path,size=size)
    img.close()

def Gtiff2rgb(image_path,outFormat='jpeg',bands=3):
    """
    Convert GeoTIFF to RGB JPEG with GDAL translate cmd tool,
    save to folder in current directory with name 'RGB'.
    This doesn't work with Osgeo gdal-py for some reason.
    
    ---- Parameters: -----
    images: List of multispectral images, relative/full path name.
    """
    image_format = os.path.split(image_path)[1].split('.')[1].lower()
    if image_format.lower() != "tiff" and image_format.lower() != "tif":
        raise Exception(f"[FORMAT ERROR] Input image needs to be of type TIFF/TIF, but is instead {image_format}")
    try:
        if bands == 3:
            command = ['gdal_translate', '-ot', 'Byte', '-of', outFormat, '-scale',\
            '-b', '1', '-b', '2', '-b', '3']
        elif bands == 1:
            command = ['gdal_translate', '-ot', 'Byte', '-of', outFormat, '-scale',\
            '-b', '1']
        elif bands == 13:
            command = ['gdal_translate', '-ot', 'Byte', '-of', outFormat, '-scale',\
            '-b', '4', '-b', '3', '-b', '2']
        else:
            raise Exception("[ERROR] Convert TIFF to JPEG/PNG: provide bands 3 (rgb), 1, or 13 (rgb output)")
        root_path = os.path.split(image_path)[0]
        out_path = root_path+'/RGB/'
        extension = '.'+outFormat
        if extension == '.jpeg': extension = '.jpg'
        out_name = os.path.split(image_path)[1].split('.')[0]+extension
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.path.exists(out_path+out_name): # Check if already exists
            command.append(image_path)
            command.append(out_path+out_name)
            subprocess.run(command)
        else:
            print('JPEG/PNG previously generated --- SKIPPING')
    except Exception as e:
        print(e)

def getFeatures(gdf):
    """ Function to parse features from GeoDataFrame in such a manner that rasterio wants them """
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

    
def reproject_raster(raster_path,crs='EPSG:4326'):
    srcRst = rasterio.open(raster_path)
    dstCrs = {'init': crs}
    
    transform,width,height = calculate_default_transform(
        srcRst.crs, dstCrs, srcRst.width, srcRst.height, *srcRst.bounds
    )
    
    kwargs = srcRst.meta.copy()
    kwargs.update({
        'crs': dstCrs,
        'transform': transform,
        'width': width,
        'height': height
    })
    
    new_name_path = raster_path.replace('.tif','_proj.tif')
    dstRst = rasterio.open(new_name_path, 'w', **kwargs)
    for i in range(1, srcRst.count + 1):
        reproject(
            source=rasterio.band(srcRst,i),
            destination=rasterio.band(dstRst,i),
            src_crs=srcRst.crs,
            dst_crs=dstCrs,
            resampling=Resampling.nearest
        )
    dstRst.close()
    srcRst.close()
    
def crop_GTiff(image_path,bbox,description):
    # Configure bbox: coordinate input: (xmin,ymin,xmax,ymax)
    # GDAL expects: (xmin,ymax,xmax,ymin)
    print(f"Cropping: {description}")
    root_path = os.path.split(image_path)[0]
    img = os.path.split(image_path)[1]
    image_format = img.split('.')[1]
    if image_format != 'tiff' and image_format != 'tif':
        raise Exception(f"[FORMAT ERROR] Input image needs to be of type .tiff / .tif, but is instead .{image_format}")

    name = description+'_'+img.split('_')[1].split('.')[0]+'_crop.tif'
    out_path = root_path+'/'+name
    if not os.path.exists(out_path):
        try:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            command = ['gdal_translate', '-of', 'GTiff', '-projwin',\
                str(xmin), str(ymax), str(xmax), str(ymin), image_path, out_path]
            subprocess.run(command)
        except Exception as e:
            print(e)
    else: print('[INFO] Raster already cropped, SKIPPING!')
    
def match_resolution(master, slave):
    master_ds = rasterio.open(master)
    master_trans = master_ds.transform
    x_res = master_trans[0] # Pixel sizes to scale to
    y_res = master_trans[4]
    
    slave_output = slave.replace('_proj_crop.tif','_projcrop_matched.tif')
    if not os.path.exists(slave_output):
        command = ['gdal_translate', '-of', 'GTiff', '-tr',\
            str(x_res), str(y_res), slave, slave_output]
        subprocess.run(command)
    

