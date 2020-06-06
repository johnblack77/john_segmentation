'''
生成scannet数据集的.sens下载路径
'''


#!usr/bin/python
# -*- coding: utf-8 -*-
str1 = "/Volumes/JohnBlack/data/scannet/scans/"
str2 = "http://kaldir.vc.in.tum.de/scannet/v1/scans/"
str3 = "/Users/john/PycharmProjects/trail/ScanNet-master/data/list.txt"
import cv2
import os
import numpy as np
list = []
for root ,dirs, files in os.walk(str1):
   for t in dirs:
       list.append(str2+t+'/'+t+'.sens')
file = open(str3,'a')
for i in range(len(list)):
    s = str(list[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
    s = s.replace("'", '').replace(',', '') + '\n'+ '\n'  # 去除单引号，逗号，每行末尾追加换行符
    file.write(s)
file.close()
print("保存文件成功")