import rasterio
import os
import numpy as np
from matplotlib import pyplot as plt
from rasterio.plot import show

def press(event):
    print('press', event.key)
    if event.key == 'c':
        too_cloudy.append(image)
        plt.close(fig)
    else:
        plt.close(fig)
        
too_cloudy = []
root_img = 'images/'
images = [root_img+f+'/'+f+'_rgb.tiff' for f in os.listdir(root_img) if not f.startswith('.')]

for image in images:
    img = rasterio.open(image).read()
    img = np.rollaxis(img,0,3)
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event',press)
    ax.imshow(img)
    plt.show()
    
print(too_cloudy)
file = open('too_inaccurate.txt', 'w')
for img in too_cloudy:
    file.write(img+'\n')
file.close()