import os
import shutil
from sklearn.model_selection import train_test_split


# Define train/val split
train_size = 0.85 # 85%

# Define images too cloudy to make the cut (should automate this if time)
# Other images omitted are temporarily stored here as well
too_cloudy = [
    "images/norway/2019-05-02:2019-06-02",
    "images/norway/2020-07-01:2020-07-31",
    "images/sudan/2019-12-01:2020-01-01",
    "images/egypt/2019-01-01:2019-01-31"
]

image_dir_norway = []
image_dir_egypt = []
image_dir_sudan = []
root_images = "images/"
for region in os.listdir(root_images):
    path = root_images + region
    if region == 'norway':
        image_dir_norway += [path+'/'+f+'/tiled_images/' for f in os.listdir(path) if not f.startswith('.')\
            and path+'/'+f not in too_cloudy]
    elif region == 'egypt':
        image_dir_egypt += [path+'/'+f+'/tiled_images/' for f in os.listdir(path) if not f.startswith('.')\
            and path+'/'+f not in too_cloudy]
    elif region == 'sudan':
        image_dir_sudan += [path+'/'+f+'/tiled_images/' for f in os.listdir(path) if not f.startswith('.')\
            and path+'/'+f not in too_cloudy]
        
path_images_norway = []
for directory in image_dir_norway:
    for tile in os.listdir(directory):
        if tile.endswith(".tif"): path_images_norway.append(directory+tile)
path_images_egypt = []
for directory in image_dir_egypt:
    for tile in os.listdir(directory):
        if tile.endswith(".tif"): path_images_egypt.append(directory+tile)
path_images_sudan = []
for directory in image_dir_sudan:
    for tile in os.listdir(directory):
        if tile.endswith(".tif"): path_images_sudan.append(directory+tile)

path_masks_norway = []
path_masks_egypt = []
path_masks_sudan = []
root_masks = "masks/"
for region in os.listdir(root_masks):
    path = root_masks + region + '/tiled_masks/'
    if region == 'norway':
        path_masks_norway += [path+f for f in os.listdir(path) if f.endswith('.tif')]
    elif region == 'egypt':
        path_masks_egypt += [path+f for f in os.listdir(path) if f.endswith('.tif')]
    elif region == 'sudan':
        path_masks_sudan += [path+f for f in os.listdir(path) if f.endswith('.tif')]        
        
temporary_norway_masks = path_masks_norway
temporary_egypt_masks = path_masks_egypt
temporary_sudan_masks = path_masks_sudan

for i in range(len(image_dir_norway)-1):
    path_masks_norway = path_masks_norway + temporary_norway_masks
for i in range(len(image_dir_egypt)-1):
    path_masks_egypt = path_masks_egypt + temporary_egypt_masks
for i in range(len(image_dir_sudan)-1):
    path_masks_sudan = path_masks_sudan + temporary_sudan_masks

path_images = path_images_norway + path_images_egypt + path_images_sudan
path_masks = path_masks_norway + path_masks_egypt + path_masks_sudan

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
        shutil.copyfile(img,train_masks_dst+str(count)+extension)
        count += 1
    count = 1
    for mask in val_masks:
        shutil.copyfile(img,val_masks_dst+str(count)+extension)
        count += 1
    
    


