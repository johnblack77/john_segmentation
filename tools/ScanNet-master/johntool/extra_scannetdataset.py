'''
提取scannet  ply文件
'''
import os
import shutil


datapath = '/Volumes/JohnBlack/data/scannet/scans_test/'
ro = '/Users/john/PycharmProjects/trail/ScanNet-master/data/test/'
li = []

for root, dirs, file in os.walk(datapath):
    if len(dirs) > 0:
        li = dirs
        for t in li:
            os.makedirs(ro + t)
    if len(file) > 0:
        for t in file:
            if 's.ply' == t[-5:]:
                shutil.copy(root + '/' + t, ro + root[43:])
            if '2.ply' == t[-5:]:
                shutil.copy(root + '/' + t, ro + root[43:])
            if 'txt'   == t[-3:]:
                shutil.copy(root + '/' + t, ro + root[43:])

