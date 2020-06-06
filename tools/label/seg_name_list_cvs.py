'''
导出SUNRGBD 语义name list .cvs
'''
import os
import scipy.io as sio
import numpy as np
import csv

str1 = '/Users/john/PycharmProjects/trail/label/data/SUNRGBD/xtion/xtion_align_data/'

stack = []
for root, dirs, files in os.walk(str1):
    if 'seg.mat' in files:
        image = root + '/image/'
        seg = root + '/seg.mat'

        data = sio.loadmat(seg)
        seg1 = data['seglabel']
        label = data['names']
        label = np.array(label)
        label = label.reshape(1,-1)
        # print(len(label[0]))
        le = len(label[0])
        for i in range(le):
            tmp = str(label[0][i])[2:-2]
            if tmp not in stack:
                stack.append(tmp)
print(len(stack))
with open('/Users/john/PycharmProjects/trail/fixlabel/data/xtion_align_data.csv','w')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(stack)



