import os
import subprocess
import json
from itertools import product

import rasterio
from rasterio.mask import mask
from rasterio import windows
from shapely.geometry import box
from fiona.crs import from_epsg
import geopandas as gpd
import pycrs

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
    if data == 'image':
        out_path = os.path.split(image_path)[0]+'/tiled_images/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    elif data == 'mask':
        out_path = os.path.split(image_path)[0]+'/tiled_masks/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    tile_write(img,output_path=out_path)

def Gtiff2rgb(image_path,outFormat='jpeg'):
    """
    Convert GeoTIFF to RGB JPEG with GDAL translate cmd tool,
    save to folder in current directory with name 'RGB'.
    
    ---- Parameters: -----
    images: List of multispectral images, relative/full path name.
    """
    image_format = os.path.split(image_path)[1].split('.')[1].lower()
    if image_format.lower() != "tiff" and image_format.lower() != "tif":
        raise Exception(f"[FORMAT ERROR] Input image needs to be of type TIFF/TIF, but is instead {image_format}")
    try:
        command = ['gdal_translate', '-ot', 'Byte', '-of', outFormat, '-scale',\
        '-b', '1', '-b', '2', '-b', '3']
        root_path = os.path.split(image_path)[0]
        out_path = root_path+'/RGB/'
        out_name = os.path.split(image_path)[1].split('.')[0]+'.jpg'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.path.exists(out_path+out_name): # Check if already exists
            command.append(image_path)
            command.append(out_path+out_name)
            subprocess.run(command)
        else:
            print('JPEG previously generated --- SKIPPING')
    except Exception as e:
        print(e)

def getFeatures(gdf):
    """ Function to parse features from GeoDataFrame in such a manner that rasterio wants them """
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def crop_GTiff(image_path,bbox):
    bbox = box(bbox[0],bbox[1],bbox[2],bbox[3])
    root_path = os.path.split(image_path)[0]
    img = os.path.split(image_path)[1]
    image_format = img.split('.')[1]
    if image_format != 'tiff' and image_format != 'tif':
        raise Exception(f"[FORMAT ERROR] Input image needs to be of type .tiff / .tif, but is instead .{image_format}")

    out_name = img.split('.')[0]+'_crop.'+img.split('.')[1]
    out_path = root_path + '/' + out_name
    if os.path.exists(out_path):
        raise Exception("[INFO] Image already cropped, SKIPPING!")

    try:
        data = rasterio.open(image_path)
        geo = gpd.GeoDataFrame({'geometry':bbox},index=[0],crs=from_epsg(4326))
        geo = geo.to_crs(crs=data.crs.data)
        coords = getFeatures(geo)

        # Crop image:
        out_img, out_transform = mask(data,shapes=coords,crop=True)

        # Modify metadata:
        out_meta = data.meta.copy()
        epsg_code = int(data.crs.data['init'][5:])
        out_meta.update({
            'driver': 'GTiff',
            'height': out_img.shape[1],
            'width': out_img.shape[2],
            'transform': out_transform,
            'crs': pycrs.parse.from_epsg_code(epsg_code).to_proj4()
        })
        with rasterio.open(out_path, 'w', **out_meta) as dest:
            print(f'Writing to {out_path}')
            dest.write(out_img)

        out_img = rasterio.open(image_path)
        meta = out_img.meta
        print(meta)

    except Exception as e:
        print(e)





