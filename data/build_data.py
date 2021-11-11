import os
import shutil
from sklearn.model_selection import train_test_split


# Define train/val split
train_size = 0.85 # 85%

# Define images too cloudy to make the cut (should automate this if time)
too_cloudy = [
    "images/2019-05-02:2019-06-02",
    "images/2020-07-01:2020-07-31"
]

root_images = "images/"
image_dirs = [root_images+f+"/tiled_images/" for f in os.listdir(root_images) if not f.startswith('.')\
    and root_images+f not in too_cloudy]
path_images = []
for directory in image_dirs:
    for tile in os.listdir(directory):
        if tile.endswith(".tif"): path_images.append(directory+tile)

root_masks = "masks/tiled_masks/"
path_temp_masks = [root_masks + f for f in os.listdir(root_masks) if f.endswith(".tif")]
path_masks = path_temp_masks
for i in range(len(image_dirs)-1):
    path_masks = path_masks + path_temp_masks
    
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
    
    


