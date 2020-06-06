# --*-- coding:utf-8 --*--
import numpy as np

'''
getCameraParam: get the camera matrix
colOrZ: color or depth
'''
def getCameraParam(colorOrZ='color', intris=[]):
    with open(intris, "r") as f:  # 打开文件
        data = f.read()  # 读取文件
       # print(data)
        data = data.split()
        if colorOrZ == 'color':
            fx_rgb = float(data[0])
            fy_rgb = float(data[2])
            cx_rgb = float(data[4])
            cy_rgb = float(data[5])
            # fx_rgb = 5.1885790117450188e+02
            # fy_rgb = 5.1946961112127485e+02
            # cx_rgb = 3.2558244941119034e+02
            # cy_rgb = 2.5373616633400465e+02
            C = np.array([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])
        else:
            fx_d = 5.8262448167737955e+02
            fy_d = 5.8269103270988637e+02
            cx_d = 3.1304475870804731e+02
            cy_d = 2.3844389626620386e+02
            C = np.array([[fx_d, 0, cx_d], [0, fy_d, cy_d], [0, 0, 1]])
    return C


def getCameraParamscannet(colorOrZ='color', intris=[]):
    with open(intris, "r") as f:  # 打开文件
        data = f.read()  # 读取文件
       # print(data)
        data = data.split()
        if colorOrZ == 'color':
            fx_rgb = float(data[0])
            fy_rgb = float(data[2])
            cx_rgb = float(data[5])
            cy_rgb = float(data[6])
            # fx_rgb = 5.1885790117450188e+02
            # fy_rgb = 5.1946961112127485e+02
            # cx_rgb = 3.2558244941119034e+02
            # cy_rgb = 2.5373616633400465e+02
            C = np.array([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])
        else:
            fx_d = 5.8262448167737955e+02
            fy_d = 5.8269103270988637e+02
            cx_d = 3.1304475870804731e+02
            cy_d = 2.3844389626620386e+02
            C = np.array([[fx_d, 0, cx_d], [0, fy_d, cy_d], [0, 0, 1]])
    return C