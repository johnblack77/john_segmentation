import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
import time


str1 = '/Users/john/PycharmProjects/trail/ScanNet-master/data/scannet_frames_25k'

list = []
for root, dirs, files in os.walk(str1):
    if 'label' == root[-5:] :
        for f in files:
            t = root + '/' + f
            data = cv2.imread(t,cv2.IMREAD_UNCHANGED)
            h, w = np.shape(data)
            for i in range(h):
                for j in range(w):
                    if data[i][j] not in list:
                        list.append(data[i][j])
                        print(list)
            break
list.sort()
print(len(list))