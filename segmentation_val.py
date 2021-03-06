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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

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
    masked_image = mask #cv2.addWeighted(image, 0.5, mask, 0.5, 0)

    fig, axs = plt.subplots(1, 3, figsize=(20,20))
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[1].imshow(mask)
    axs[1].set_title('Segmentation Mask')
    #predimg = cv2.addWeighted(imgs[i]/255, 0.6, _p, 0.4, 0)
    axs[2].imshow(masked_image)
    axs[2].set_title('Masked Image')
    plt.show()

    model = keras.models.load_model("segmentation_model_sat-2-5")
    model.summary()

    max_show = 1
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
        plt.imshow(_p)
        plt.axis("off")
        plt.subplot(132)
        plt.title("Original")
        plt.imshow(_s)
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

    print(f'preds: {pred[1]}')
    print(f'segs: {segs[1]}')

    #pred1 = np.argmax(pred[1], axis=-1)
    #print(f'pred1: {pred1}')
    #seg1 = np.argmax(segs[1], axis=-1)
    #print(f'seg1: {seg1}')
    predictions = []
    segmentations = []
    for i in range(len(pred)):
        predictions.append(np.argmax(pred[i], axis=-1))
        segmentations.append(np.argmax(segs[i], axis=-1))

    print(f'preds: {predictions[1]}')
    print(f'segs: {segmentations[1]}')

    segmentations = np.array(segmentations)
    predictions = np.array(predictions)

    print(f"segmentation shape: {segmentations.shape}")
    print(f"predictions shape: {predictions.shape}")

    pred1D = predictions.reshape(-1)
    segs1D = segmentations.reshape(-1)

    print(f"segmentation 1d: {segs1D.shape}")
    print(f"predictions 1d: {pred1D.shape}")

    print(f"Confusion matrix: \n {tf.math.confusion_matrix(segs1D, pred1D, num_classes=N_CLASSES+1)}")

    global_precision = precision_score(segs1D, pred1D, average='micro')
    global_recall = recall_score(segs1D, pred1D, average='micro')
    print(f'Global Precision score: {global_precision}')
    print(f'Global Recall score: {global_recall}')
    print(f'Global F1 score: {(2*global_precision*global_recall)/(global_recall+global_precision)}')
    
    macro_precision = precision_score(segs1D, pred1D, average='macro')
    macro_recall = recall_score(segs1D, pred1D, average='macro')
    print(f'macro Precision score: {macro_precision}')
    print(f'macro Recall score: {macro_recall}')
    print(f'macro F1 score: {(2*macro_precision*macro_recall)/(macro_recall+macro_precision)}')
    f1_macro = f1_score(segs1D, pred1D, average='macro')
    print(f1_macro)

    weighted_precision = precision_score(segs1D, pred1D, average='weighted')
    weighted_recall = recall_score(segs1D, pred1D, average='weighted')
    print(f'Weighted Precision score: {weighted_precision}')
    print(f'Weighted Recall score: {weighted_recall}')
    print(f'Weighted F1 score: {(2*weighted_precision*weighted_recall)/(weighted_recall+weighted_precision)}')
    #print(f"Confusion matrix: \n {confusion_matrix(segs1D, pred1D)}")
    f1=f1_score(segs1D, pred1D, average='weighted')
    print(f'F1 score: {f1}')
