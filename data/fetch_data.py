import os
import datetime
from re import X

from sentinelhub import SHConfig
from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions

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

if __name__ == '__main__':
    make_request = False # SET TO TRUE IF REQUESTING NEW DOWNLOAD, IT WILL NOT CHECK BY ITSELF IF DIRECTORY NAMES ARE CHANGED, THUS THIS IS NECESSARY TO AVOID DUPLICATES
    name_by_location = True # SET TO TRUE IF RENAIMING DOWNLOADED FILES AND FOLDERS TO DATE INTERVAL OF CAPTURED IMAGE BY SENTINEL.

    if make_request:
        # Instantiate SH-instance parameters
        config = SHConfig()
        # Configure instance
        config.instance_id = '86154f73-e694-4f78-84e1-c40c7ee96ed6'
        config.sh_client_id = '74d33a87-245f-4def-ae3e-e2535e7d814c'
        config.sh_client_secret = 'K7ngSx@,0+-k7lGqb5I(+X3&6HY5H;pO0Zx,VS@F'

        # Image parameters
        
        # Set image coordinates
        coords_wgs84_melhus = [10.194969,63.154200,10.371094,63.233937]
        x_diff = round(coords_wgs84_melhus[2] - coords_wgs84_melhus[0], 6)
        y_diff = round(coords_wgs84_melhus[3] - coords_wgs84_melhus[1], 6)
        
        coords_wgs84_otta = [8.261719,61.834603]
        coords_wgs84_orkla = [9.722471,63.179190]
        coords_wgs84_rauma = [7.626228,62.499004]
        coords_wgs84_jostedal = [6.681747,61.690523]
        coords_wgs84_selbu = [11.074219,63.184882]
        coords_wgs84_elverum = [11.338234,60.923926]
        coords_wgs84_støren = [10.242348,63.012847]
        coords_wgs84_fonna = [6.206760,59.922248]
        coords_wgs84_nidelven = [10.402336,63.272719]
        coords_wgs84_sunndal = [8.525219,62.643000]
        coords_wgs84_hovin = [10.196171,63.068491]
        coords_wgs84_glomma = [11.193352,59.444115]
        
        coords = [coords_wgs84_orkla,coords_wgs84_otta,coords_wgs84_rauma,coords_wgs84_jostedal,
                  coords_wgs84_selbu,coords_wgs84_elverum,coords_wgs84_støren,coords_wgs84_fonna,
                  coords_wgs84_nidelven,coords_wgs84_sunndal,coords_wgs84_hovin,coords_wgs84_glomma]
        
        for coord in coords:
            coord += [round(coord[0]+x_diff,6),round(coord[1]+y_diff,6)]
        
        # Set bounds of images (bbox)
        bboxes = []
        bbox_sizes = []
        resolution = 10
        for coord in coords:
            box = BBox(bbox=coord,crs=CRS.WGS84)
            box_size = bbox_to_dimensions(box,resolution=resolution)
            bboxes.append(box)
            bbox_sizes.append(box_size)
        
        # Set image time intervals (done manually to find best possible images for each region)
        time_intervals = [
            ('2019-07-01','2019-07-31'), ('2019-07-01','2019-07-31'),
            ('2019-07-01','2019-07-31'), ('2019-08-01','2019-08-31'),
            ('2019-07-01','2019-07-31'), ('2020-07-01','2020-07-31'),
            ('2019-07-01','2019-07-31'), ('2019-07-01','2019-07-31'),
            ('2019-07-01','2019-07-31'), ('2019-07-01','2019-07-31'),
            ('2019-07-01','2019-07-31'), ('2019-07-01','2019-07-31'),
            ('2020-06-01','2020-06-30'), ('2020-07-01','2020-07-31'),
            ('2020-07-01','2020-07-31'), ('2020-08-01','2020-08-31'),
            ('2020-06-01','2020-06-30'), ('2020-06-01','2020-06-30'),
            ('2020-06-01','2020-06-30'), ('2020-07-01','2020-07-31'),
            ('2020-06-01','2020-06-30'), ('2020-08-01','2020-08-31'),
            ('2020-06-01','2020-06-30'), ('2020-06-01','2020-06-30'),
            ]
    
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
          
        requests = []    
        for idx, t in enumerate(time_intervals):
            if idx > 11: idx = idx - 12
            request = get_all_band_request(t,bboxes[idx],bbox_sizes[idx],evalscript_all_bands,config)
            requests.append(request.download_list[0])
        data = SentinelHubDownloadClient(config=config).download(requests,max_threads=5)
    
    if name_by_location:
        import json
        
        coords_to_loc = {
            '[8.261719, 61.834603, 8.437844, 61.91434]': 'otta',
            '[9.722471, 63.17919, 9.898596, 63.258927]': 'orkla',
            '[7.626228, 62.499004, 7.802353, 62.578741]': 'rauma',
            '[6.681747, 61.690523, 6.857872, 61.77026]': 'jostedal',
            '[11.338234, 60.923926, 11.514359, 61.003663]': 'elverum',
            '[11.074219, 63.184882, 11.250344, 63.264619]': 'selbu',
            '[10.242348, 63.012847, 10.418473, 63.092584]': 'støren',
            '[6.20676, 59.922248, 6.382885, 60.001985]': 'fonna',
            '[10.402336, 63.272719, 10.578461, 63.352456]': 'nidelven',
            '[8.525219, 62.643, 8.701344, 62.722737]': 'sunndal',
            '[10.196171, 63.068491, 10.372296, 63.148228]': 'hovin',
            '[11.193352, 59.444115, 11.369477, 59.523852]': 'glomma'
        }
        root_path = 'images/'
        image_locations = [root_path+f for f in os.listdir(root_path) if not f.startswith('.')]
        for data in image_locations:
            files = [data+'/'+f for f in os.listdir(data) if not f.startswith('.')]
            for file in files:
                if file.endswith('.json'): # Find date range and location in .json request
                    with open(file,'r') as j:
                        request_info = json.loads(j.read())
                    date_interval = request_info['payload']['input']['data'][0]['dataFilter']['timeRange']
                    from_date = date_interval['from'].split('T')[0]
                    to_date = date_interval['to'].split('T')[0]
                    date_interval = from_date + ':' + to_date
                    
                    bounds = request_info["payload"]["input"]["bounds"]["bbox"]
                    loc = coords_to_loc[str(bounds)]
                    new_name = loc + '_' + date_interval
            for file in files:
                # Rename content files:
                format_type = '.'+os.path.split(file)[1].split('.')[1]
                new_path = data + '/' + new_name + format_type
                os.rename(file,new_path)
            # Rename containing directories:
            new_root = os.path.split(data)[0] + '/' + new_name
            os.rename(data,new_root)
            

                
