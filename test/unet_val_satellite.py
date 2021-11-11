
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

import PIL.Image as Image

import datetime

import os
import seaborn as sns
import cv2
from keras.callbacks import ModelCheckpoint

EPOCHS=10
BATCH_SIZE=10
HEIGHT=64
WIDTH=64
N_CLASSES=7

def LoadImage(name, path):
    print(name)
    if not name.startswith('.'):
        img = Image.open(os.path.join(path+'/images/', name))
        img = np.array(img)

        m = Image.open(os.path.join(path+'/masks/', name))
        m = np.array(m)

        image = img
        mask = m

        return image, mask


def bin_image(mask):
    #bins = np.array([[0:2], [175:177], [229:231], [107:109], [203:205], [64:67], [254:255]])
    bins = np.array([0, 176, 230, 108, 204, 66, 255])
    bins.sort()
    # [Water, Crops, Built area, Grass, Scrub, Trees, Bare Ground]
    new_mask = np.digitize(mask, bins)
    return new_mask

def getSegmentationArr(image, classes, width=WIDTH, height=HEIGHT):
    seg_labels = np.zeros((height, width, classes))
    img = image[:, :]

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

def DataGenerator(path, batch_size=BATCH_SIZE, classes=N_CLASSES):
    files = os.listdir(path+'/images')
    while True:
        for i in range(0, len(files), batch_size):
            batch_files = files[i : i+batch_size]
            imgs=[]
            segs=[]
            for file in batch_files:
                if not file.startswith('.'):
                    image, mask = LoadImage(file, path)
                    mask_binned = bin_image(mask)
                    labels = getSegmentationArr(mask_binned, classes)

                    imgs.append(image)
                    segs.append(labels)
            yield np.array(imgs), np.array(segs)


classes = 7

train_folder = "data/Train"
valid_folder = "data/Val"

num_of_training_samples = len(os.listdir(train_folder))
num_of_valid_samples = len(os.listdir(valid_folder))

train_gen = DataGenerator(train_folder, batch_size=BATCH_SIZE)
val_gen = DataGenerator(valid_folder, batch_size=BATCH_SIZE)

imgs, segs = next(train_gen)
print(imgs.shape, segs.shape)

image = imgs[9]
mask = give_color_to_seg_img(np.argmax(segs[9], axis=-1))
masked_image = cv2.addWeighted(image/255, 0.5, mask, 0.5, 0)

fig, axs = plt.subplots(1, 3, figsize=(20,20))
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[1].imshow(mask)
axs[1].set_title('Segmentation Mask')
#predimg = cv2.addWeighted(imgs[i]/255, 0.6, _p, 0.4, 0)
axs[2].imshow(masked_image)
axs[2].set_title('Masked Image')
plt.show()

model = keras.models.load_model("segmentation_model_sat")
model.summary()

max_show = 10
imgs, segs = next(val_gen)
pred = model.predict(imgs)

for i in range(max_show):
    _p = give_color_to_seg_img(np.argmax(pred[i], axis=-1))
    _s = give_color_to_seg_img(np.argmax(segs[i], axis=-1))

    predimg = cv2.addWeighted(imgs[i]/255, 0.5, _p, 0.5, 0)
    trueimg = cv2.addWeighted(imgs[i]/255, 0.5, _s, 0.5, 0)

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.title("Prediction")
    plt.imshow(predimg)
    plt.axis("off")
    plt.subplot(122)
    plt.title("Original")
    plt.imshow(trueimg)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("pred_"+str(i)+".png", dpi=150)
    plt.show()

"""
https://www.kaggle.com/ashishsingh226/semantic-segmentation-cityscapes
"""
