import tensorflow as tf
from tensorflow.python.ops.array_ops import batch_gather
from tensorflow import keras

#model = tf.saved_model.load("model", tags=[])

path = "model"
batch_size = 100
input_shape = [214,214,3]
inputs = [batch_size]

"""

model = tf.keras.Model(...)
tf.saved_model.save(model, path)
imported = tf.saved_model.load(path)
outputs = imported(inputs)

model.compile()

"""
#model = keras.models.load_model(path)

model = tf.saved_model.load(
    path, tags=None, options=None
)

print("MobileNet has {} trainable variables: {}, ...".format(
          len(model.trainable_variables),
          ", ".join([v.name for v in model.trainable_variables[:5]])))

#model.summary()