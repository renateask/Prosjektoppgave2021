import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds

import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import tensorflow_hub as hub

import PIL.Image as Image

#print(tfds.list_builders())

ds, info = tfds.load('eurosat/all', split='train', with_info=True)
assert isinstance(ds, tf.data.Dataset)
print(ds)

print("\n")

print(info)

frame = tfds.as_dataframe(ds.take(4), info)

print(frame)

""" 

example = ds.take(1)

print(f"example: {example}")

fig = tfds.show_examples(ds, info)
plt.show()

img = plt.imread(example.filename())
plt.imshow(img)
plt.show()

"""

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



model = keras.Sequential([
    layers.BatchNormalization(),
    layers.Conv2D(filters=16, kernel_size=5, padding='same', activation='elu'),
    layers.MaxPool2D(),
    
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, padding='same', activation='elu'),
    layers.Conv2D(32, 3, padding='same', activation='elu'),
    layers.MaxPool2D(),
    
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding='same', activation='elu'),
    layers.Conv2D(64, 3, padding='same', activation='elu'),
    layers.MaxPool2D(),
    
    layers.Flatten(),
    layers.Dense(128, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])



model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)


EPOCHS = 10
STEPS_PER_EPOCH = int(info.splits['train'].num_examples * 0.7) // BATCH_SIZE

early_stopping = tf.keras.callbacks.EarlyStopping(patience=7, min_delta=0.001, restore_best_weights=True)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[early_stopping],
)

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].plot()
plt.show()

IMAGE_SHAPE = (64, 64)

river = Image.open('River_1005.jpg').resize(IMAGE_SHAPE)

img = plt.imread("River_1005.jpg")
plt.imshow(img)
plt.show()

river = np.array(river)/255.0
print(river.shape)

result = model.predict(river[np.newaxis, ...])
print(result.shape)

model.save("test_model")