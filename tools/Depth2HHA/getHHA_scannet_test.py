# --*-- coding:utf-8 --*--
# for sunrgbd
import math
import cv2
import os
import math

import shutil
import scipy.io as sio

from utils.rgbd_util import *
from utils.getCameraParam import *

str1 = '/Users/john/PycharmProjects/trail/label/data/scannet_frames_test'

'''
must use 'COLOR_BGR2GRAY' here, or you will get a different gray-value with what MATLAB gets.
'''
def getImage(root='demo'):
    D = cv2.imread(os.path.join(root, '0.png'), cv2.COLOR_BGR2GRAY)/10000
    RD = cv2.imread(os.path.join(root, '0_raw.png'), cv2.COLOR_BGR2GRAY)/10000
    return D, RD


'''
C: Camera matrix
D: Depth image, the unit of each element in it is "meter"
RD: Raw depth image, the unit of each element in it is "meter"
'''
def getHHA(C, D, RD):
    missingMask = (RD == 0);
    pc, N, yDir, h, pcRot, NRot = processDepthImage(D * 100, missingMask, C);

    tmp = np.multiply(N, yDir)
    acosValue = np.minimum(1,np.maximum(-1,np.sum(tmp, axis=2)))
    angle = np.array([math.degrees(math.acos(x)) for x in acosValue.flatten()])
    angle = np.reshape(angle, h.shape)

    '''
    Must convert nan to 180 as the MATLAB program actually does. 
    Or we will get a HHA image whose border region is different
    with that of MATLAB program's output.
    '''
    angle[np.isnan(angle)] = 180        


    pc[:,:,2] = np.maximum(pc[:,:,2], 100)
    I = np.zeros(pc.shape)

    # opencv-python save the picture in BGR order.
    I[:,:,2] = 31000/pc[:,:,2]
    I[:,:,1] = h
    I[:,:,0] = (angle + 128-90)

    # print(np.isnan(angle))

    '''
    np.uint8 seems to use 'floor', but in matlab, it seems to use 'round'.
    So I convert it to integer myself.
    '''
    I = np.rint(I)

    # np.uint8: 256->1, but in MATLAB, uint8: 256->255
    I[I>255] = 255
    HHA = I.astype(np.uint8)
    return HHA

if __name__ == "__main__":

    c = 0
    D = list
    for root, dirs, files in os.walk(str1):
        if 'depth' in dirs:
            xx = []
            yy = []
            for _,_,xx in os.walk(root + '/depth'):
                pass
            if 'hha' in dirs:
                for _,_,yy in os.walk(root + '/hha'):
                    pass

            if len(xx) == len(yy):
                c+=len(xx)
                print(c)
            else:
                if len(yy) >0:
                    shutil.rmtree(root + '/hha',True)

                depth = root + '/depth/'
                intrin = root + '/intrinsics_depth.txt'
                camera_matrix = getCameraParamscannet('color', intrin)

                os.makedirs(root + '/hha')

                for r, d, f in os.walk(depth):
                    for t in f:
                        D = cv2.imread(r + t, cv2.COLOR_BGR2GRAY) / 750
                        hha = getHHA(camera_matrix, D, D)
                        cv2.imwrite(root + '/hha/' + t, hha)
                        c += 1
                        print(c)






    
    ''' multi-peocessing example '''
    '''
    from multiprocessing import Pool
    
    def generate_hha(i):
        # generate hha for the i-th image
        return
    
    processNum = 16
    pool = Pool(processNum)

    for i in range(img_num):
        print(i)
        pool.apply_async(generate_hha, args=(i,))
        pool.close()
        pool.join()
    ''' 
