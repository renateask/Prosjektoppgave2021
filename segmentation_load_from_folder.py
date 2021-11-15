
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

def give_color_to_seg_img(seg, n_classes=N_CLASSES):

    seg_img = np.zeros( (seg.shape[1],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)

    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)

print(f"\n Label dictionary: {CLASSES}\n")

model = keras.models.load_model("segmentation_model_sat2")

files = os.listdir("RGB")

print(files)
files.sort()
print(files)

names = []
for i in range(183):
    names.append(str(i+1)+'.jpg')

print(names, len(names))
IMAGE_SHAPE = (64, 64)

predictions = []

fig = plt.figure()
plt.title("847cc28c918f50fcdebf46a90c340ac6")
fig.tight_layout(pad=50)

preds = []
for i in range(len(names)):
    if names[i] != ".DS_Store":
        if names[i] != "122.jpg":
            path = "RGB/"+names[i]
            tile = Image.open(path)
            tile = np.array(tile)/255.0
            print(f"\nTile shape: {tile.shape}")
            result = model.predict(tile[np.newaxis, ...])


            #print(f"Prediction of: {names[i]} is {result}")

            pred = np.argmax(result[0], axis=-1)
            preds.append(pred)
            print(f'pred: {pred.shape}')

            _p = give_color_to_seg_img(pred)

            #decode = CLASSES[predicted_class.numpy()]
            #print(f"Decoded prediction {predicted_class}, {decode} \n")


            plt.subplot(14,14,i+1)
            #img = plt.imread(path)
            plt.imshow(_p)
            plt.title(f"Prediction: of Filename: {names[i]}")
plt.show()

predictions = np.array(preds)

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

print(f'shape of predictions: {predictions.shape}')
for i in range(14):
    one.append(give_color_to_seg_img(predictions[i]))
    two.append(give_color_to_seg_img(predictions[i+14]))
    three.append(give_color_to_seg_img(predictions[i+28]))
    four.append(give_color_to_seg_img(predictions[i+42]))
    five.append(give_color_to_seg_img(predictions[i+56]))
    six.append(give_color_to_seg_img(predictions[i+70]))
    seven.append(give_color_to_seg_img(predictions[i+84]))
    eight.append(give_color_to_seg_img(predictions[i+98]))
    nine.append(give_color_to_seg_img(predictions[i+112]))
    ten.append(give_color_to_seg_img(predictions[i+126]))
    eleven.append(give_color_to_seg_img(predictions[i+140]))
    twelve.append(give_color_to_seg_img(predictions[i+154]))
    thirteen.append(give_color_to_seg_img(predictions[i+168]))


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

image = np.concatenate((one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen), axis=1)
#print(f'shape of two: {two.shape}\n, two {two}')

plt.imshow(image)
plt.show()