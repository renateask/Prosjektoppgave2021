
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


if __name__ == '__main__':
    train_folder = "data/train"
    valid_folder = "data/validation"

    num_training_samples = len(os.listdir(train_folder+'/images'))
    num_valid_samples = len(os.listdir(valid_folder+'/images'))

    train_gen = DataGenerator(train_folder, batch_size=BATCH_SIZE)
    val_gen = DataGenerator(valid_folder, batch_size=BATCH_SIZE)

    imgs, segs = next(train_gen)

    image = imgs[7]
    mask = give_color_to_seg_img(np.argmax(segs[7], axis=-1))
    masked_image = cv2.addWeighted(image, 0.5, mask, 0.5, 0)

    fig, axs = plt.subplots(1, 3, figsize=(20,20))
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[1].imshow(mask)
    axs[1].set_title('Segmentation Mask')
    #predimg = cv2.addWeighted(imgs[i]/255, 0.6, _p, 0.4, 0)
    axs[2].imshow(masked_image)
    axs[2].set_title('Masked Image')
    plt.show()
    
    model = keras.models.load_model("segmentation_model_sat2")
    model.summary()

    max_show = 10
    imgs, segs = next(val_gen)
    pred = model.predict(imgs)

    for i in range(max_show):
        _p = give_color_to_seg_img(np.argmax(pred[i], axis=-1))
        _s = give_color_to_seg_img(np.argmax(segs[i], axis=-1))
        _i = imgs[i]

        predimg = cv2.addWeighted(imgs[i]/255, 0.5, _p, 0.5, 0)
        trueimg = cv2.addWeighted(imgs[i]/255, 0.5, _s, 0.5, 0)

        plt.figure(figsize=(12,6))
        plt.subplot(131)
        plt.title("Prediction")
        plt.imshow(predimg)
        plt.axis("off")
        plt.subplot(132)
        plt.title("Original")
        plt.imshow(trueimg)
        plt.axis("off")
        plt.subplot(133)
        plt.title("Original")
        plt.imshow(_i)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("pred_"+str(i)+".png", dpi=150)
        plt.show()

    """
    https://www.kaggle.com/ashishsingh226/semantic-segmentation-cityscapes
    """
