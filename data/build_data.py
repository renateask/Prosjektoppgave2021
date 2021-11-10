import os
import shutil

# Paths to images and masks

image_dirs = [
    "images/2020-05-01:2020-06-01/",
    "images/2019-09-01:2019-10-01/",
    "images/2019-10-01:2019-11-01/",
    "images/2019-07-02:2019-08-01/",
    "images/2019-08-01:2019-09-01/",
    "images/2020-10-31:2020-11-30/",
    "images/2020-09-30:2020-10-31/",
]

n_imgs = len(image_dirs)

for i in range(len(image_dirs)):
    image_dirs[i] = image_dirs[i] + 'tiled_images/RGB/'

images =[]
for image in image_dirs:
    tiles = [image + tile for tile in os.listdir(image) if tile.endswith('.jpg')]
    for tile in tiles:
        images.append(tile)
        
mask_dirs = "masks/tiled_masks/RGB/"  

mask_single = []
for mask in os.listdir(mask_dirs):
    if mask.endswith('.jpg'):
        mask_single.append(mask_dirs + mask)

masks = []
for i in range(n_imgs):
    masks += mask_single
        
data_dir = "data/"
if not os.path.exists(data_dir):
    image_dir = data_dir + 'images/'
    mask_dir = data_dir + 'masks/'
    os.makedirs(image_dir)
    os.makedirs(mask_dir)
    count = 0
    for image in images:
        shutil.copyfile(image,image_dir+str(count)+'.jpg')
        count += 1
    count = 0
    for mask in masks:
        shutil.copyfile(mask,mask_dir+str(count)+'.jpg')
        count += 1
    
