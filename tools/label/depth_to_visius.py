import cv2
import os.path
import glob
import numpy as np
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
pngfile = '/Users/john/PycharmProjects/trail/label/data/scannet_frames_25k/scene0000_01/depth/000000.png'
path = '/Users/john/PycharmProjects/trail/label/data/scannet_frames_25k/scene0000_01/color/000000.jpg'

# READ THE DEPTH
im_depth = cv2.imread(pngfile,cv2.IMREAD_UNCHANGED)
img = cv2.imread(path,cv2.IMREAD_UNCHANGED)
# apply colormap on deoth image(image must be converted to 8-bit per pixel first)
plt.figure()
plt.subplot(1, 2, 1)
plt.title('img')
plt.imshow(im_depth)
plt.subplot(1, 2, 2)
plt.title('gsd')
plt.imshow(img)
plt.show()

print('jdx')