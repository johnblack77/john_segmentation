'''import os
path = '/home/jhon/Documents/john_segmentation/data/CamVid/val'
f = open('/home/jhon/Documents/john_segmentation/data/CamVid/val.txt','a')
for root, dirs, files in os.walk(path):
    for i in files:
        f.write(path+'/'+i +' '+path+'annot/'+i+'\n')
f.close()
print('f')'''

import os
import cv2
import numpy as np
path = '/home/jhon/Documents/john_segmentation/data/CamVid/test.txt'
f = open(path,'r')

for i in f.readlines():
    c = i.strip().split(' ')
    #print(c[0], c[1])
    img =cv2.imread(c[0],cv2.IMREAD_UNCHANGED)
    h,w,cc=np.shape(img)
    if h!=304 or w!=304 or cc!=3:
        print(c[0])
    img = cv2.imread(c[1], cv2.IMREAD_UNCHANGED)
    h, w = np.shape(img)
    if h != 304 or w != 304 :
        print(c[1])
print('ff')