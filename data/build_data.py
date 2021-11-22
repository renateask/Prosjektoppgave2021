import os
import shutil
from sklearn.model_selection import train_test_split


# Define train/val split
train_size = 0.85 # 85%

esri_innacurate = [
    'elverum_2020-10-01:2020-10-31', 'elverum_2019-07-01:2019-07-31', 'elverum_2019-10-01:2019-10-31',
    'elverum_2017-07-01:2017-07-31', 'fonna_2021-07-01:2021-07-31', 'fonna_2021-08-01:2021-08-31',
    'fonna_2020-07-01:2020-07-31', 'fonna_2019-07-01:2019-07-31', 'fonna_2019-08-01:2019-08-31',
    'fonna_2018-08-01:2018-08-31', 'fonna_2018-07-01:2018-07-31', 'fonna_2017-08-01:2017-08-31',
    'hovin_2021-09-01:2021-09-30', 'hovin_2020-09-01:2020-09-30', 'nidelven_2019-06-01:2019-06-30',
    'nidelven_2018-08-01:2018-08-31', 'orkla_2019-06-01:2019-06-30', 'orkla_2018-08-01:2018-08-31',
    'otta_2021-09-01:2021-09-30', 'otta_2020-09-01:2020-09-30', 'otta_2019-07-01:2019-07-31',
    'otta_2019-09-01:2019-09-30', 'otta_2017-09-01:2017-09-30', 'otta_2018-09-01:2018-09-30',
    'rauma_2017-10-01:2017-10-31', 'rauma_2018-10-01:2018-10-31', 'rauma_2019-10-01:2019-10-31',
    'rauma_2020-06-01:2020-06-30', 'rauma_2020-10-01:2020-10-31', 'rauma_2021-06-01:2021-06-30',
    'rauma_2021-10-01:2021-10-31', 'selbu_2017-06-01:2017-06-30', 'selbu_2019-06-01:2019-06-30',
    'støren_2018-09-01:2018-09-30', 'støren_2021-09-01:2021-09-30', 'trondheim_2019-06-01:2019-06-30'
]

root_images = "images/"
image_dirs = [root_images+f+"/tiled_images/" for f in os.listdir(root_images) if not f.startswith('.') and\
    '_test_' not in f and f not in esri_innacurate]
order_of_regions = [region.split('/')[1].split('_')[0] for region in image_dirs]
path_images = []

for directory in image_dirs:
    for tile in os.listdir(directory):
        if tile.endswith(".tif") and tile.startswith('tile'): path_images.append(directory+tile)

root_masks = "masks/"
mask_dirs = []
path_masks = []
for region in order_of_regions:
    dir = root_masks+'tiled_masks_'+region+'/'
    mask_dirs.append(dir)
for directory in mask_dirs:
    for tile in os.listdir(directory):
        if tile.endswith(".tif") and tile.startswith('tile'): path_masks.append(directory+tile) 

# Check if order matches:
for img, mask in zip(path_images,path_masks):
    region_img = img.split('/')[1].split('_')[0]
    region_mask = mask.split('/')[1].split('_')[2]
    tile_img = os.path.split(img)[1]
    tile_mask = os.path.split(mask)[1]
    if not region_img == region_mask:
        raise Exception("Order of regions of masks does not correspond with that of images.")
    if not tile_img == tile_mask:
        raise Exception("Order of tiles of masks does not correspond with that of images.")


# Build dataset:
train_imgs_dst = "train/images/"
train_masks_dst = "train/masks/"
val_imgs_dst = "validation/images/"
val_masks_dst = "validation/masks/"

# Train/Val Split:
train_imgs,val_imgs,train_masks,val_masks = train_test_split(path_images,path_masks,train_size=train_size)
extension = '.'+os.path.split(train_imgs[0])[1].split('.')[1]
if not os.path.exists(train_imgs_dst) and not os.path.exists(train_masks_dst)\
    and not os.path.exists(val_imgs_dst) and not os.path.exists(val_masks_dst):
    # Create directories and populate if not already done
    os.makedirs(train_imgs_dst)
    os.makedirs(train_masks_dst)
    os.makedirs(val_imgs_dst)
    os.makedirs(val_masks_dst)
    
    count = 1
    for img in train_imgs:
        shutil.copyfile(img,train_imgs_dst+str(count)+extension)
        count += 1
    count = 1
    for img in val_imgs:
        shutil.copyfile(img,val_imgs_dst+str(count)+extension)
        count += 1
    count = 1
    for mask in train_masks:
        shutil.copyfile(mask,train_masks_dst+str(count)+extension)
        count += 1
    count = 1
    for mask in val_masks:
        shutil.copyfile(mask,val_masks_dst+str(count)+extension)
        count += 1
    
    


