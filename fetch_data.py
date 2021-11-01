import os
import datetime
import numpy as np
import subprocess
import time
from itertools import product

import rasterio
from rasterio.plot import show
from rasterio import windows
from osgeo import gdal

from sentinelhub import SHConfig
from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions, DownloadRequest

"""
This script fetches data from Sentinel-hub. All bands of the MSI-instrument included.
Data is saved to folder ~/images/. Parameters for hub-config-access, as well as wgs84-box-coordinates,
resolution, dates to fetch and save, etc. may be altered in the top part of the script.
"""

def get_all_band_request(time_interval,box,box_size,eval,config):
    """
    Request Sentinel-hub images.

    ----- Parameters: -----
    time_interval (tuple) :: (start_date, end_date)
    box (BBox object) :: Define BBox from sentinelhub, with valid coordinates and coordinate system
    box_size :: Size of BBox object, defined by resolution and box
    eval (JS evalscript) :: String object, define which bands to request and what to return
    config (json) :: SHConfig() instance, with valid credentials

    ----- Returns: -----
    Hub-request of least cloudy image from Sentinel 2 L1C in given time period, for given coordinate-box
    """
    return SentinelHubRequest(
        data_folder = 'images',
        evalscript=eval,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=time_interval,
                mosaicking_order='leastCC'
                )
        ],
        responses=[
            SentinelHubRequest.output_response('default',MimeType.TIFF)
        ],
        bbox=box,
        size=box_size,
        config=config
    )

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

def tile_and_save(path):
    """
    Read Geotiff image from sentinelhub, append image to list.

    ----- Parameters: -----
    path (str) :: location of image folders
    bands (tuple) :: which multispectral image bands to read (default: RGB)
    tile (bool) :: if True --> tile image and save image tiles
    """
    loc = [f for f in os.listdir(path) if not f.startswith('.')]
    images = []
    for l in loc:
        img_path = path+l+'/response.tiff'
        img = rasterio.open(img_path)
        images.append(img)

    for img,l in zip(images,loc):
        out_path = path+l+'/tiled_images/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            tile_write(img,output_path=out_path)
        else:
            print("Images already tiled, please remove 'tiled_images' folder if retiling, this to save space!")
            break

def GeoTIFF_2_RGBJPEG(image_names,outFormat='jpeg'):
    """
    Convert GeoTIFF to RGB JPEG with GDAL translate cmd tool,
    save to folder in current directory with name 'RGB'.
    
    ---- Parameters: -----
    images: List of multispectral images, relative/full path name.
    """
    for img in image_names:
        command = ['gdal_translate', '-ot', 'Byte', '-of', outFormat, '-scale',\
        '-b', '4', '-b', '3', '-b', '2']
        root_path = os.path.split(img)[0]
        out_path = root_path+'/RGB/'
        out_name = os.path.split(img)[1].split('.')[0]+'.jpg'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.path.exists(out_path+out_name): # Check if already exists
            command.append(img)
            command.append(out_path+out_name)
            subprocess.run(command)
        else:
            print('JPEG previously generated --- SKIPPING')
        


if __name__ == '__main__':
    # Instantiate SH-instance parameters
    config = SHConfig()
    # Configure instance
    config.instance_id = '86154f73-e694-4f78-84e1-c40c7ee96ed6'
    config.sh_client_id = '74d33a87-245f-4def-ae3e-e2535e7d814c'
    config.sh_client_secret = 'K7ngSx@,0+-k7lGqb5I(+X3&6HY5H;pO0Zx,VS@F'

    # Image parameters
    coords_wgs84 = [10.194969,63.154200,10.371094,63.233937]
    resolution = 10
    start_time = datetime.datetime(2020,1,1)
    end_time = datetime.datetime(2020,12,31)
    n_img = 13 # Monthly images

    box = BBox(bbox=coords_wgs84,crs=CRS.WGS84)
    box_size = bbox_to_dimensions(box,resolution=resolution)
    tdelta = (end_time-start_time)/n_img
    edges = [(start_time + i*tdelta).date().isoformat() for i in range(n_img)]
    time_intervals = [(edges[i], edges[i+1]) for i in range(len(edges)-1)]

    evalscript_all_bands = """
        //VERSION=3
        function setup(){
            return {
                input: [{
                    bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B09","B10","B11","B12"]
                }],
                output: {
                    bands: 13
                }
            };
        }
        function evaluatePixel(sample) {
            return [sample.B01,sample.B02,
                    sample.B03,sample.B04,
                    sample.B05,sample.B06,
                    sample.B07,sample.B08,
                    sample.B09,sample.B10,
                    sample.B11,sample.B12];
        }
    """

    # Download images from sentinelhub
    requests = [get_all_band_request(t,box,box_size,evalscript_all_bands,config) for t in time_intervals]
    requests = [request.download_list[0] for request in requests]
    data = SentinelHubDownloadClient(config=config).download(requests,max_threads=5)


    # Tile images and save?
    tile = False
    if tile:
        path = 'images/'
        tile_and_save(path)
    
    # Create JPG images for image and/or tiles?
    convert_full_raster = False
    convert_tiled_raster = False

    root_path = 'images/'
    tiled_images = []
    full_rasters = []

    if convert_full_raster:
        full_images = [root_path + f for f in os.listdir(root_path) if not f.startswith('.')]
        for img in full_images:
            full_rasters.append(img+'/response.tiff')
        GeoTIFF_2_RGBJPEG(full_rasters)
    print("[INFO] Preparing to convert tiled rasters to JPEG (RGB).")
    time.sleep(3)
    if convert_tiled_raster:
        full_images = [root_path + f for f in os.listdir(root_path) if not f.startswith('.')]
        full_images = [f + '/tiled_images/' for f in full_images]
        for img in full_images:
            tiles = [tile for tile in os.listdir(img) if not tile.startswith('.')]
            for tile in tiles:
                tiled_images.append(img+tile)
        GeoTIFF_2_RGBJPEG(tiled_images)
    
