
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import PIL.Image as Image


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
import keras as keras
import keras.layers as layers
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
import tensorflow_hub as hub
import rasterio
import datetime
import os
import seaborn as sns
import cv2
from keras.callbacks import ModelCheckpoint

EPOCHS=30
BATCH_SIZE=10
HEIGHT=64
WIDTH=64
CLASSES = {
    1: 'Water',
    2: 'Trees',
    3: 'Grass',
    4: 'Flooded Vegetation',
    5: 'Crops',
    6: 'Scrub/Shrub',
    7: 'Built Area',
    8: 'Bare Ground',
    9: 'Snow/Ice'
}
N_CLASSES=len(CLASSES)


def LoadImage(name, path):
    """ Return: (h,w,n)-np.arrays """
    # Images to np-arrays
    path = os.path.join(path, name)
    image_arr = rasterio.open(path).read()
    # Convert dimensions to standard (n,height,width) --> (height,width,n)
    image = np.rollaxis(image_arr,0,3)
    return image


def bin_image(mask):
    print(CLASSES.keys())
    bins = np.array([pixel_val for pixel_val in CLASSES.keys()])
    new_mask = np.digitize(mask, bins)
    return new_mask

def getSegmentationArr(image, classes, width=WIDTH, height=HEIGHT):
    seg_labels = np.zeros((height, width, classes))
    img = image[:, :, 0]

    for c in range(classes):
        seg_labels[:, :, c] = (img == c ).astype(int)
    return seg_labels

def give_color_to_seg_img(seg, n_classes=N_CLASSES):

    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)

    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)

def DataLoader(path):
    imgs=[]
    names = []
    for i in range(182):
        names.append(str(i+1)+'.tif')
    print(names, len(names))

    for name in names:
        if not name.startswith('.'):
            image = LoadImage(name, path)
            imgs.append(image)
            #print(image)
    return np.array(imgs)

print(f"\n Label dictionary: {CLASSES}\n")

model = keras.models.load_model("segmentation_model_sat-2-2")          ########## Segmentation Model
model.summary()

path = "data_4/validation/images"

num_tiles = len(os.listdir(path))

imgs = DataLoader(path)
#print('imgs', imgs)
image = imgs[7]


plt.imshow(image)
plt.title('Original Image')
plt.show()


IMAGE_SHAPE = (64, 64)

predictions = []

fig = plt.figure()
plt.title("path")
fig.tight_layout(pad=50)

#print(f'shape of imgs: {imgs.shape}')
#print(f'shape of imgs[0]: {imgs[0].shape}')

preds = []
for i in range(len(imgs)):
    tile = imgs[i]
    #print(f'shape of tile: {tile.shape}')
    result = model.predict(tile[np.newaxis, ...])
    pred = np.argmax(result[0], axis=-1)
    preds.append(pred)
    #print(f'pred: {pred.shape}')

    _p = give_color_to_seg_img(pred)

    plt.subplot(14,14,i+1)
    plt.imshow(_p)
    plt.title(f"Prediction: of tile: {i}")
plt.show()

predictions = np.array(preds)

plt.imshow(give_color_to_seg_img(predictions[25]))
plt.title('predictions[25]')
plt.show()


one = []
two = []
three = []
four = []
five = []
six = []
seven = []
eight = []
nine = []
ten = []
eleven = []
twelve = []
thirteen = []
fourteen = []

N = 14

# Plots 13x13 tiles

print(f'shape of predictions: {predictions.shape}')
for i in range(N):
    one.append(give_color_to_seg_img(predictions[i]))
    two.append(give_color_to_seg_img(predictions[i+N]))
    three.append(give_color_to_seg_img(predictions[i+N*2]))
    four.append(give_color_to_seg_img(predictions[i+N*3]))
    five.append(give_color_to_seg_img(predictions[i+N*4]))
    six.append(give_color_to_seg_img(predictions[i+N*5]))
    seven.append(give_color_to_seg_img(predictions[i+N*6]))
    eight.append(give_color_to_seg_img(predictions[i+N*7]))
    nine.append(give_color_to_seg_img(predictions[i+N*8]))
    ten.append(give_color_to_seg_img(predictions[i+N*9]))
    eleven.append(give_color_to_seg_img(predictions[i+N*10]))
    twelve.append(give_color_to_seg_img(predictions[i+N*11]))
    thirteen.append(give_color_to_seg_img(predictions[i+N*12]))
    #thirteen.append(give_color_to_seg_img(predictions[i+N*13]))
    #if i == 182:
    #    i = i-1
    #fourteen.append(give_color_to_seg_img(predictions[i+N*14]))


#two = np.concatenate((give_color_to_seg_img(preds[0]),give_color_to_seg_img(preds[1]), give_color_to_seg_img(preds[2]), give_color_to_seg_img(preds[4]), give_color_to_seg_img(preds[5])), axis=0)
one = np.concatenate((one), axis=0)
two = np.concatenate((two), axis=0)
three = np.concatenate((three), axis=0)
four = np.concatenate((four), axis=0)
five = np.concatenate((five), axis=0)
six = np.concatenate((six), axis=0)
seven = np.concatenate((seven), axis=0)
eight = np.concatenate((eight), axis=0)
nine = np.concatenate((nine), axis=0)
ten = np.concatenate((ten), axis=0)
eleven = np.concatenate((eleven), axis=0)
twelve = np.concatenate((twelve), axis=0)
thirteen = np.concatenate((thirteen), axis=0)
#fourteen = np.concatenate((fourteen), axis=0)

print(f'shape of 1: {one.shape}\n, two {N}')
print(f'shape of 2: {two.shape}\n, two {N}')
print(f'shape of 3: {three.shape}\n, two {N}')
print(f'shape of 4: {four.shape}\n, two {N}')
print(f'shape of 5: {five.shape}\n, two {N}')
print(f'shape of 6: {six.shape}\n, two {N}')
print(f'shape of 7: {seven.shape}\n, two {N}')
print(f'shape of 8: {eight.shape}\n, two {N}')
print(f'shape of 9: {nine.shape}\n, two {N}')
print(f'shape of 10: {ten.shape}\n, two {N}')
print(f'shape of 11: {eleven.shape}\n, two {N}')
print(f'shape of 12: {twelve.shape}\n, two {N}')
print(f'shape of 13: {thirteen.shape}\n, two {N}')


image = np.concatenate((one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen), axis=1)


plt.imshow(image)
plt.title(f"Prediction of image: {path}")
plt.show()

################# putting togeter the satelite image

one = []
two = []
three = []
four = []
five = []
six = []
seven = []
eight = []
nine = []
ten = []
eleven = []
twelve = []
thirteen = []

print(f'shape of imgs: {imgs.shape}')
for i in range(N):
    one.append((imgs[i]))
    two.append((imgs[i+N]))
    three.append((imgs[i+N*2]))
    four.append((imgs[i+N*3]))
    five.append((imgs[i+N*4]))
    six.append((imgs[i+N*5]))
    seven.append((imgs[i+N*6]))
    eight.append((imgs[i+N*7]))
    nine.append((imgs[i+N*8]))
    ten.append((imgs[i+N*9]))
    eleven.append((imgs[i+N*10]))
    twelve.append((imgs[i+N*11]))
    thirteen.append((imgs[i+N*12]))

one = np.concatenate((one), axis=0)
two = np.concatenate((two), axis=0)
three = np.concatenate((three), axis=0)
four = np.concatenate((four), axis=0)
five = np.concatenate((five), axis=0)
six = np.concatenate((six), axis=0)
seven = np.concatenate((seven), axis=0)
eight = np.concatenate((eight), axis=0)
nine = np.concatenate((nine), axis=0)
ten = np.concatenate((ten), axis=0)
eleven = np.concatenate((eleven), axis=0)
twelve = np.concatenate((twelve), axis=0)
thirteen = np.concatenate((thirteen), axis=0)

sat_image = np.concatenate((one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen), axis=1)

fig, axs = plt.subplots(1, 2, figsize=(20,10))
axs[0].imshow(image)
axs[0].set_title('Segmented Image')
axs[1].imshow(sat_image)
axs[1].set_title('Satellite Image')
plt.show()