import os
import datetime

from sentinelhub import SHConfig
from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions

"""
This script fetches data from Sentinel-hub. All bands of the MSI-instrument included.
Data is saved to folder ~/images/. Parameters for hub-config-access, as well as wgs84-box-coordinates,
resolution, dates to fetch and save, etc. may be altered in the top part of the script.
"""

def get_all_band_request(time_interval,box,box_size,eval,config,country):
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
        data_folder = 'images/'+country,
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

if __name__ == '__main__':
    make_request = False # SET TO TRUE IF REQUESTING NEW DOWNLOAD, IT WILL NOT CHECK BY ITSELF IF DIRECTORY NAMES ARE CHANGED, THUS THIS IS NECESSARY TO AVOID DUPLICATES
    name_by_dates = False # SET TO TRUE IF RENAIMING DOWNLOADED FILES AND FOLDERS TO DATE INTERVAL OF CAPTURED IMAGE BY SENTINEL.

    if make_request:
        # Instantiate SH-instance parameters
        config = SHConfig()
        # Configure instance
        config.instance_id = '86154f73-e694-4f78-84e1-c40c7ee96ed6'
        config.sh_client_id = '74d33a87-245f-4def-ae3e-e2535e7d814c'
        config.sh_client_secret = 'K7ngSx@,0+-k7lGqb5I(+X3&6HY5H;pO0Zx,VS@F'

        # Image parameters
        coords_wgs84_norway = [10.194969,63.154200,10.371094,63.233937]
        coords_wgs84_egypt = [32.395248,30.137705,32.580986,30.311542]
        coords_wgs84_sudan = [31.301079,21.753441,31.451454,21.925848]
        resolution = 10
        start_time = datetime.datetime(2019,1,1)
        end_time = datetime.datetime(2020,12,31)
        n_months = 24 # Monthly images
        n_img = 13

        box_norway = BBox(bbox=coords_wgs84_norway,crs=CRS.WGS84)
        box_size_norway = bbox_to_dimensions(box_norway,resolution=resolution)
        box_egypt = BBox(bbox=coords_wgs84_egypt,crs=CRS.WGS84)
        box_size_egypt = bbox_to_dimensions(box_egypt,resolution=resolution)
        box_sudan = BBox(bbox=coords_wgs84_sudan,crs=CRS.WGS84)
        box_size_sudan = bbox_to_dimensions(box_sudan,resolution=resolution)
        tdelta = (end_time-start_time)/n_months
        edges = [(start_time + i*tdelta).date().isoformat() for i in range(n_months+1)]
        time_intervals_nor = []
        time_intervals_egt = []
        time_intervals_sud = []
        for i in range(len(edges)-1): # Only include non-snowy images in Norway
            if not edges[i].split('-')[1] in ['01','02','03','04','11','12']:
                time_intervals_nor.append((edges[i],edges[i+1]))
            elif edges[i].split('-')[1] in ['01','02'] and edges[i].split('-')[0] == '2019':
                time_intervals_egt.append((edges[i],edges[i+1]))
            elif edges[i].split('-')[1] in ['11','12'] and edges[i].split('-')[0] == '2019':
                time_intervals_sud.append((edges[i],edges[i+1]))

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
                if (sample.CLM == 1) {
                    return [sample.B01,sample.B02,
                        sample.B03, 0.75 + sample.B04,
                        sample.B05,sample.B06,
                        sample.B07,sample.B08,
                        sample.B09,sample.B10,
                        sample.B11,sample.B12]
                }
                return [sample.B01*2.5,sample.B02*2.5,
                        sample.B03*2.5,sample.B04*2.5,
                        sample.B05*2.5,sample.B06*2.5,
                        sample.B07*2.5,sample.B08*2.5,
                        sample.B09*2.5,sample.B10*2.5,
                        sample.B11*2.5,sample.B12*2.5];
            }
        """

        # Download images from sentinelhub
        if not os.path.exists('images/'):
            os.makedirs('images')
        
        # Download images of Trøndelag-region, Norway
        if not os.path.exists('images/norway'):
            os.makedirs('images/norway')
            requests = [get_all_band_request(t,box_norway,box_size_norway,evalscript_all_bands,config,country='norway') for t in time_intervals_nor]
            requests = [request.download_list[0] for request in requests]
            data = SentinelHubDownloadClient(config=config).download(requests,max_threads=5)
        
        # Download images from top of Red-Sea
        if not os.path.exists('images/egypt'):
            os.makedirs('images/egypt')
            requests = [get_all_band_request(t,box_egypt,box_size_egypt,evalscript_all_bands,config,country='egypt') for t in time_intervals_egt]
            requests = [request.download_list[0] for request in requests]
            data = SentinelHubDownloadClient(config=config).download(requests,max_threads=5)
            
        if not os.path.exists('images/sudan'):
            os.makedirs('images/sudan')
            requests = [get_all_band_request(t,box_sudan,box_size_sudan,evalscript_all_bands,config,country='sudan') for t in time_intervals_sud]
            requests = [request.download_list[0] for request in requests]
            data = SentinelHubDownloadClient(config=config).download(requests,max_threads=5)
    
    if name_by_dates:
        import json
        root_path = 'images/'
        region_paths = [root_path+f for f in os.listdir(root_path) if not f.startswith('.')]
        image_locations = []
        for region in region_paths:
            for f in os.listdir(region):
                if not f.startswith('.'):
                    image_locations.append(region+'/'+f)
        for data in image_locations:
            files = [data+'/'+f for f in os.listdir(data) if not f.startswith('.')]
            for file in files:
                if file.endswith('.json'): # Find date range in .json request
                    with open(file,'r') as j:
                        request_info = json.loads(j.read())
                    date_interval = request_info['payload']['input']['data'][0]['dataFilter']['timeRange']
                    from_date = date_interval['from'].split('T')[0]
                    to_date = date_interval['to'].split('T')[0]
                    new_name = from_date + ':' + to_date
            for file in files:
                # Rename content files:
                format_type = '.'+os.path.split(file)[1].split('.')[1]
                new_path = data + '/' + new_name + format_type
                os.rename(file,new_path)
            # Rename containing directories:
            new_root = os.path.split(data)[0] + '/' + new_name
            os.rename(data,new_root)
                
                

