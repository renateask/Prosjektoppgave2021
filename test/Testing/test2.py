import tensorflow as tf
import tensorflow_hub as hub


hub_url = "https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1"

"""
hub_layer = hub.KerasLayer(hub_url, input_shape=[])
#hub_layer(train_examples[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(19))

model.summary()
"""

module = hub.Module(hub_url)

module_spec = hub.load_module_spec(module)
height, width = hub.get_expected_image_size(module_spec)

print(height, width)

images = [100, height, width, 3]  # A batch of images with shape [batch_size, height, width, 3].
module = hub.Module(module_spec)
features = module(images)   # A batch with shape [batch_size, num_features].