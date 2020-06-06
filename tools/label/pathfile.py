'''

从原始SUNRGBD数据集提取文件
新增seg.png
生成：
0
    image
    depth
    depth_bfx
    inistri
    seg.mat
1
...
10335

'''

import os
import shutil
import scipy.io as sio
import cv2

str1 = '/Users/john/PycharmProjects/trail/label/data/SUNRGBD/'
str2 = '/Users/john/PycharmProjects/trail/label/mum/'

c = 0

for root, dirs, files in os.walk(str1):
    if 'seg.mat' in files:

            dir = ''
            depth = root + dir + '/' + 'depth/'
            depth_bfx = root + dir + '/' + 'depth_bfx/'
            image = root + dir + '/' + 'image/'
            intrin = root + dir + '/' + 'intrinsics.txt'
            seg = root + dir + '/' + 'seg.mat'
            c += 1
            os.makedirs(str2 + str(c))

            for r, d, f in os.walk(depth):
                shutil.copy(r + f[0], str2 + str(c) + '/' + str(c) + '_d.png')
            for r, d, f in os.walk(depth_bfx):
                shutil.copy(r + f[0], str2 + str(c) + '/' + str(c) + '_d_bfx.png')
            for r, d, f in os.walk(image):
                shutil.copy(r + f[0], str2 + str(c) + '/' + str(c) + '_rgb.jpg')
            shutil.copy(intrin, str2 + str(c) + '/' + str(c) + '_intris.txt')
            shutil.copy(seg, str2 + str(c) + '/' + str(c) + '_seg.mat')

            file = open(str2 + str(c) + '/flag.txt', 'w')
            file.write('flag')
            file.close()

            print(c)