import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
import time
import h5py

path_root = '/Users/john/PycharmProjects/trail/label/data/'
path_seg = '/Users/john/PycharmProjects/trail/label/data/SUNRGBD2Dseg.mat'
path_meta = '/Users/john/PycharmProjects/trail/label/data/SUNRGBDMeta.mat'

count = 0
start = time.time()

seg_mat = h5py.File(path_seg)
t1 = seg_mat['SUNRGBD2Dseg']
t2 = t1['seglabel']

meta = sio.loadmat(path_meta)
meta = meta['SUNRGBDMeta']


for seg_num in range(len(t2)):
    path_save = str(meta[0][seg_num][0])[2:-2]
    path_save = path_root + path_save + '/seg.png'
    arrseg = np.array(t1[(t2[seg_num][0])],dtype='uint8').T
    cv2.imwrite(path_save,arrseg)
    print(seg_num + 1)