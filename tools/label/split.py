import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
import time

splitpath = '/Users/john/PycharmProjects/trail/label/data/allsplit.mat'
rpath = ''
tpath = ''

data = sio.loadmat(splitpath)
data = data['alltrain']