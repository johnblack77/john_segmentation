'''
删除SUNRGBD 无用文件
'''
import os
import shutil


datapath = '/Users/john/PycharmProjects/trail/label/data/SUNRGBD'

for root, dirs, file in os.walk(datapath):
    if 'seg.mat' in file:
        shutil.rmtree(root + '/annotation', True)
        shutil.rmtree(root + '/annotation2D3D', True)
        shutil.rmtree(root + '/annotation2Dfinal', True)
        shutil.rmtree(root + '/annotation3D', True)
        shutil.rmtree(root + '/annotation3Dfinal', True)
        shutil.rmtree(root + '/annotation3Dlayout', True)
        shutil.rmtree(root + '/depth', True)
        shutil.rmtree(root + '/extrinsics', True)
        shutil.rmtree(root + '/fullres', True)
        os.remove(root + '/intrinsics.txt')
        os.remove(root + '/scene.txt')
        os.remove(root + '/seg.mat')
        os.remove(root + '/seg1.png')
