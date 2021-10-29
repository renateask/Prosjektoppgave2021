


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds

import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import tensorflow_hub as hub

import PIL.Image as Image


ds, info = tfds.load('eurosat/rgb',
                        with_info=True,
                        split='train')

tfds.show_examples(ds, info)

print(info)

NUM_CLASSES = info.features['label'].num_classes
lables = info.features['label']
print(f"number of classes: {NUM_CLASSES}, names {lables}")

frame = tfds.as_dataframe(ds.take(5), info)

print(frame)


BATCH_SIZE = 16
AUTO = tf.data.experimental.AUTOTUNE
SHUFFLE_BUFFER = int(info.splits['train'].num_examples * 0.7)

ds_train, ds_valid = tfds.load('eurosat/rgb',
                               split=['train[:70%]', 'train[70%:]'],
                               as_supervised=True)

def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

ds_train = (ds_train
            .map(preprocess, AUTO)
            .cache()
            .shuffle(SHUFFLE_BUFFER)
            .repeat()
            # Augmentations go here .map(augment, AUTO)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(AUTO))

ds_valid = (ds_valid
            .map(preprocess, AUTO)
            .cache()
            .batch(BATCH_SIZE)
            .prefetch(AUTO))

NUM_CLASSES = info.features['label'].num_classes


model = keras.models.load_model("eurosat_classifier")


test_pred = model.predict(ds_valid)
test_pred = np.argmax(test_pred, axis=1)
print(f"Prediction: {test_pred}, shape: {test_pred.shape}")

val_evaluate = model.evaluate(ds_valid,return_dict=True)

print(val_evaluate)

IMAGE_SHAPE = (64, 64)

river = Image.open('River_1005.jpg').resize(IMAGE_SHAPE)

img = plt.imread("River_1005.jpg")
plt.imshow(img)
plt.show()

river = np.array(river)#/255.0
print(river.shape)

result = model.predict(river[np.newaxis, ...])
print(result.shape)
print(result)

predicted_class = tf.math.argmax(result[0], axis=-1)
print(predicted_class)


for images, labels in ds_valid.take(1):  # only take first element of dataset
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()

print(f"lables {numpy_labels}, shape {numpy_labels.shape}")

#ds_numpy = tfds.as_numpy(ds_valid)

print("ds_valid:  ", ds_valid)

def get_labels_from_tfdataset(tfdataset, batched=False):

    labels = list(map(lambda x: x[1], tfdataset)) # Get labels

    if not batched:
        return tf.concat(labels, axis=0) # concat the list of batched labels

    return labels

labels = get_labels_from_tfdataset(ds_valid, batched=True)

#print(labels)

label = []

for e in range(len(labels)):
    for i in labels[e]:
        label.append(i.numpy())



#print(f"Lables: {label}, shape: {len(label)}")

print(f"Confusion matrix: \n {tf.math.confusion_matrix(label, test_pred)}")

tf.keras.utils.plot_model(model, show_shapes=True)
plt.show()

model.summary()

"""
{'AnnualCrop': 0,
 'Forest': 1,
 'HerbaceousVegetation': 2,
 'Highway': 3,
 'Industrial': 4,
 'Pasture': 5,
 'PermanentCrop': 6,
 'Residential': 7,
 'River': 8,
 'SeaLake': 9}

Confusion matrix:
 [[795   1   0  13   1   8  24   0  31   5]
 [  3 837   6   2   0   6   1   0   0   2]
 [  3   8 799  14   4  15  59  21   7   3]
 [ 32   3  13 517  31  16  39  23  70   0]
 [  1   0   3   2 701   0   6  27   4   0]
 [ 12  13  11   9   0 543  15   1  13   1]
 [ 13   0  47  28  23   3 661   4   3   0]
 [  0   0   2   2  14   0   1 886   0   0]
 [ 12   4   7  38   2   8   9   2 663   1]
 [  3   3   2   0   0   3   0   0   9 873]]

{'loss': 0.4619160294532776, 'sparse_categorical_accuracy': 0.8981481194496155}

"""
