
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

#"https://tfhub.dev/google/remote_sensing/eurosat-resnet50/1"
#"https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1"

hub_url = "https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1"
hub_layer = hub.KerasLayer(hub_url, trainable=True, input_shape=[])

IMAGE_SHAPE = (64, 64)

classifier = tf.keras.Sequential([
    layers.InputLayer(input_shape=IMAGE_SHAPE + (3,)),
    hub.KerasLayer(hub_url, input_shape=IMAGE_SHAPE+(3,)),
    layers.Dropout(rate=0.2),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

classifier.summary()

classifier.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)


EPOCHS = 10
STEPS_PER_EPOCH = int(info.splits['train'].num_examples * 0.7) // BATCH_SIZE

early_stopping = tf.keras.callbacks.EarlyStopping(patience=7, min_delta=0.001, restore_best_weights=True)

history = classifier.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[early_stopping],
)

classifier.save("eurosat_classifier2")

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].plot()
print(history_frame)
plt.show()


#river = tf.keras.utils.get_file('River_1005.jpg')
#
#
river = Image.open('River_1005.jpg').resize(IMAGE_SHAPE)

img = plt.imread("River_1005.jpg")
plt.imshow(img)
plt.show()

river = np.array(river)/255.0
print(river.shape)

result = classifier.predict(river[np.newaxis, ...])
print(result.shape)
print(result)

predicted_class = tf.math.argmax(result[0], axis=-1)
print(predicted_class)


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
"""
