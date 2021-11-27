
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
from tensorflow.keras.optimizers import Adam

EPOCHS1=5 #30
EPOCHS2=3 #15
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
    image_arr = rasterio.open(os.path.join(path+'/images/',name)).read()
    mask_arr = rasterio.open(os.path.join(path+'/masks/',name)).read()
    # Convert dimensions to standard (n,height,width) --> (height,width,n)
    image = np.rollaxis(image_arr,0,3)
    mask = np.rollaxis(mask_arr,0,3)

    return image, mask


def bin_image(mask):
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

def DataGenerator(path, batch_size=BATCH_SIZE, classes=N_CLASSES):
    files = os.listdir(path+'/images')
    while True:
        imgs=[]
        segs=[]
        for file in files:
            if not file.startswith('.'):
                image, mask = LoadImage(file, path)
                #mask_binned = bin_image(mask)
                #labels = getSegmentationArr(mask_binned, classes)

                imgs.append(image)
                segs.append(mask)
        return np.array(imgs), np.array(segs)


if __name__ == '__main__':
    train_folder = "data_64_no_snow/train"
    valid_folder = "data_64_no_snow/validation"
    test_folder = "data/validation"

    num_training_samples = len(os.listdir(train_folder+'/images'))
    num_valid_samples = len(os.listdir(valid_folder+'/images'))

    #train_gen = DataGenerator(train_folder, batch_size=BATCH_SIZE)
    #val_gen = DataGenerator(valid_folder, batch_size=BATCH_SIZE)

    imgs, segs = DataGenerator(train_folder, batch_size=BATCH_SIZE)

    #imgs, segs = next(train_gen)

    print(f"Segs shape: {segs.shape}")

    classes_per_image = []
    for image in segs:
        image = image.reshape(-1)
        number_of_classes = 0
        classes = {
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
            6: False,
            7: False,
            8: False,
            9: False
        }
        for pixel in image:
            classes[pixel] = True
        for key in classes:
            if classes[key] == True:
                number_of_classes += 1
        #print(f"classes: {classes}, number of classes: {number_of_classes}")
        classes_per_image.append(number_of_classes)
    classes_per_image = np.array(classes_per_image)
    print(f"classes per image: {classes_per_image}, shape: {classes_per_image.shape}, average: {np.average(classes_per_image)}")


