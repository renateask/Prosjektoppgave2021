
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import PIL.Image as Image

label_dict = {0: 'AnnualCrop',
    1: 'Forest',
    2: 'HerbaceousVegetation',
    3: 'Highway',
    4: 'Industrial',
    5: 'Pasture',
    6: 'PermanentCrop',
    7: 'Residential',
    8: 'River',
    9: 'SeaLake'}

print(f"\n Label dictionary: {label_dict}\n")

model = keras.models.load_model("eurosat_classifier32")

files = os.listdir("RGB")

print(files)
files.sort()
print(files)

IMAGE_SHAPE = (64, 64)

predictions = []

fig = plt.figure()
plt.title("847cc28c918f50fcdebf46a90c340ac6")
fig.tight_layout(pad=50)

for i in range(10):
    if files[i] != ".DS_Store":
        path = "RGB/"+files[i]

        tile = Image.open(path)
        tile = np.array(tile)/255.0
        print(f"\nTile shape: {tile.shape}")
        result = model.predict(tile[np.newaxis, ...])
        print(f"Prediction of: {files[i]} is {result}")
        predicted_class = tf.math.argmax(result[0], axis=-1)
        decode = label_dict[predicted_class.numpy()]
        print(f"Decoded prediction {predicted_class}, {decode} \n")


        plt.subplot(2,5,i+1)
        img = plt.imread(path)
        plt.imshow(img)
        plt.title(f"Prediction: {decode}\n Filename: {files[i]}")
plt.show()






#River_1005.jpg
#test2.jpg
#testbilde1-elv.jpg
#fil = "River_1005.jpg"

#river = Image.open(fil).resize(IMAGE_SHAPE)

#river = np.array(river)/255.0
#print("Image shape after resize ",river.shape)

#result = model.predict(river[np.newaxis, ...])
#print(result.shape)
#print("\n Prediction of: ",fil,"  ",result)

#predicted_class = tf.math.argmax(result[0], axis=-1)
#print(f"\n Decoded prediction {predicted_class}, {label_dict[predicted_class.numpy()]} \n")

#img = plt.imread(fil)
#plt.imshow(img)
#plt.show()




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
