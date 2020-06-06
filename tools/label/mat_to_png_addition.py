'''seg相似labelname 检查'''

'''

# stage 1

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

str1 = '/Users/john/PycharmProjects/trail/label/data/SUNRGBD/'


def gene(name, stack, color, segimg, data):
    if name in stack:
        flag = 999
        seg1 = data['seglabel']
        h, w = np.shape(seg1)
        for ind in range(len(stack)):
            if stack[ind] == name:
                flag = ind + 1

                for i in range(h):
                    for j in range(w):
                        if flag == seg1[i, j]:
                            segimg[i, j] = color
    return segimg


def showpare(name, c, img, segimg):
    if c in segimg:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('img')
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.title(name)
        plt.imshow(segimg)
        plt.show()


c = 0


c1 = 22
c2 = 55
c3 = 99



for root, dirs, files in os.walk(str1):
    if 'seg.mat' in files:
        dir = ''
        imageroot = root + dir + '/' + 'image/'
        segroot = root + dir + '/' + 'seg.mat'

        img = []
        segimg = []
        data = sio.loadmat(segroot)
        label = data['names']
        label = np.array(label)
        label = label.reshape(1, -1)
        le = len(label[0])
        stack = []
        for i in range(le):
            tmp = str(label[0][i])[2:-2]
            stack.append(tmp)

        for r, d, f in os.walk(imageroot):
            img = mpimg.imread(r + f[0])
            h, w, _ = np.shape(img)
            segimg = np.zeros((h, w))

        segimg = gene('tv', stack, c1, segimg, data)
        # segimg = gene('hall_wall', stack, c2, segimg, data)
        # segimg = gene('back_room_wall', stack, c3, segimg, data)

        c += 1
        print(c)

        showpare('c1', c1, img, segimg)
        # showpare('c3', c3, img, segimg)
'''


'''
stage 2
'''


import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# 修改项
str1 = '/Users/john/PycharmProjects/trail/label/data/SUNRGBD'
segclas = ['monitor','monitor_screen','monitor_backing','monitor_stand']



def gene(name, stack, color, segimg, data):
    if name in stack:
        seg1 = data['seglabel']
        h, w = np.shape(seg1)
        for ind in range(len(stack)):
            if stack[ind] == name:
                flag = ind + 1

                for i in range(h):
                    for j in range(w):
                        if flag == seg1[i, j]:
                            segimg[i, j] = color
    return segimg

def showpare(name, c, img, segimg):
    if c in segimg:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('img')
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.title(name)
        plt.imshow(segimg)
        plt.show()


c = 0
c1 = 20
for root, dirs, files in os.walk(str1):
    if 'seg.mat' in files:

        imageroot = root + '/image/'
        segroot = root + '/seg.mat'
        # segpng = root + dir + '/' + 'seg1.png'

        img = []
        segimg = []

        data = sio.loadmat(segroot)
        label = data['names']
        label = np.array(label)
        label = label.reshape(1, -1)
        le = len(label[0])
        stack = []
        for i in range(le):
            tmp = str(label[0][i])[2:-2]
            stack.append(tmp)

        for r, d, f in os.walk(imageroot):
            img = mpimg.imread(r + f[0])
            h, w, _ = np.shape(img)
            segimg = np.zeros((h, w))
            # plt.figure()
            # plt.subplot(1, 1, 1)
            # plt.title('img')
            # plt.imshow(img)
            # plt.show()

        c1 = 20
        for sc in segclas:
            segimg = gene(sc, stack, c1, segimg, data)
            showpare(sc, c1, img, segimg)
            c1+=20

        c += 1
        print(c)
