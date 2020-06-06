import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt


def get_label_name_colors(csv_path):
    """
    read csv_file and save as label names and colors list
    :param csv_path: csv color file path
    :return: lable name list, label color list
    """
    label_names = []
    label_colors = []
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for i, row in enumerate(reader):
            if i > 0:  # 跳过第一行
                label_names.append(row[0])
                label_colors.append([int(row[1]), int(row[2]), int(row[3])])

    return label_names, label_colors


def create_label_map(label_colors, rows, cols, row_height, col_width):
    """
    create a colorful label image for plt to annotate
    :param label_colors: label color list
    :param rows: num of figure rows
    :param cols: num of figure cols
    :param row_height: height of each row
    :param col_width: width of each col
    :return:
    """
    label_map = np.ones((row_height * rows, col_width * cols, 3), dtype='uint8') * 255
    cnt = 0
    for i in range(rows):  # 1st row is black = background
        for j in range(cols):
            if cnt >= len(label_colors):  # in case, num of lables < rows * cols
                break
            beg_pix = (i * row_height, j * col_width)
            end_pix = (beg_pix[0] + 20, beg_pix[1] + 20)  # 20 is color square side
            label_map[beg_pix[0]:end_pix[0], beg_pix[1]:end_pix[1]] = label_colors[cnt][::-1]  # RGB->BGR
            cnt += 1
    cv2.imwrite('label_map%dx%d.png' % (rows, cols), label_map)


def plt_label_map(label_names, label_colors, rows, cols, row_height, col_width, figsize=(10, 8), fig_title='color map'):
    """
    read cv2 saved colorful label image and use plt to annotate the label names
    :param label_names: lable name list
    :param label_colors: label color list
    :param rows: num of figure rows
    :param cols: num of figure cols
    :param row_height: height of each row
    :param col_width: width of each col
    :param figsize: overall figure size, like (10, 8)
    :param fig_title: figure title, like ADE20K-150class
    :return:
    """
    # create origin map
    if os.path.exists('label_map%dx%d.png' % (rows, cols)):
        os.remove('label_map%dx%d.png' % (rows, cols))
    create_label_map(label_colors, rows, cols, row_height, col_width)
    label_map = plt.imread('label_map%dx%d.png' % (rows, cols))

    # show origin map
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.title(fig_title + '\n', fontweight='black')  # 上移一段距离，哈哈
    plt.imshow(label_map)

    cnt = 0
    for i in range(rows):  # 1st row is black = background
        for j in range(cols):
            if cnt >= len(label_names):  # in case, num of lables < rows * cols
                break
            beg_pix = (j * col_width, i * row_height)  # note! (y,x)
            plt.annotate('%s' % label_names[cnt],
                         xy=beg_pix, xycoords='data', xytext=(+13, -8), textcoords='offset points',
                         color='k')
            cnt += 1

    plt.show()


if __name__ == '__main__':
    # ADE20K
    label_names, label_colors = get_label_name_colors(csv_path='ade150.csv')
    plt_label_map(label_names, label_colors, rows=10, cols=15, row_height=30, col_width=200, figsize=(22, 4), fig_title='ADE20K-150class')
    # SUN-RGBD
    label_names, label_colors = get_label_name_colors(csv_path='sun37.csv')
    plt_label_map(label_names, label_colors, rows=4, cols=10, row_height=30, col_width=200, figsize=(14, 3), fig_title='SUNRGBD-37class')
    # CamVid
    label_names, label_colors = get_label_name_colors(csv_path='camvid32.csv')
    plt_label_map(label_names, label_colors, rows=4, cols=10, row_height=30, col_width=200, figsize=(16, 3), fig_title='CamVid-32class')






import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

"""
传入 rows, cols, persize，确定制作想要的 行数 x 列数，以及每个 label 的大小（正方形宽度）
"""

label_colors = [(0, 0, 0),  # 0=background
                (148, 65, 137), (255, 116, 69), (86, 156, 137), (202, 179, 158), (155, 99, 235),
                (161, 107, 108), (133, 160, 103), (76, 152, 126), (84, 62, 35), (44, 80, 130),
                (31, 184, 157), (101, 144, 77), (23, 197, 62), (141, 168, 145), (142, 151, 136),
                (115, 201, 77), (100, 216, 255), (57, 156, 36), (88, 108, 129), (105, 129, 112),
                (42, 137, 126), (155, 108, 249), (166, 148, 143), (81, 91, 87), (100, 124, 51),
                (73, 131, 121), (157, 210, 220), (134, 181, 60), (221, 223, 147), (123, 108, 131),
                (161, 66, 179), (163, 221, 160), (31, 146, 98), (99, 121, 30), (49, 89, 240),
                (116, 108, 9), (161, 176, 169), (80, 29, 135), (177, 105, 197), (139, 110, 246)]

label_names = ['background',
               'wall', 'floor', 'cabinet', 'bed', 'chair',
               'sofa', 'table', 'door', 'window', 'bookshelf',
               'picture', 'counter', 'blinds', 'desk', 'shelves',
               'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
               'clothes', 'ceiling', 'books', 'fridge', 'tv',
               'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
               'person', 'night_stand', 'toilet', 'sink', 'lamp',
               'bathtub', 'bag', '38', '39', '40']


def create_label_map(rows, cols, persize):
    label_map = np.zeros((persize * rows, persize * cols, 3), dtype='uint8')
    cnt = 1
    for i in range(1, rows):  # 1st row is black = background
        for j in range(cols):
            beg_pix = (i * persize, j * persize)
            end_pix = (beg_pix[0] + persize, beg_pix[1] + persize)
            label_map[beg_pix[0]:end_pix[0], beg_pix[1]:end_pix[1]] = label_colors[cnt][::-1]
            cnt += 1
    cv2.imwrite('label_map%dx%dx%d.png' % (rows, cols, persize), label_map)


def plt_label_map(rows, cols, persize):
    # create origin map
    if not os.path.exists('label_map%dx%dx%d.png' % (rows, cols, persize)):
        create_label_map(rows, cols, persize)

    label_map = plt.imread('label_map%dx%dx%d.png' % (rows, cols, persize))

    # show origin map
    plt.figure(figsize=(8, 6))
    plt.imshow(label_map)

    # text at label[0]
    plt.annotate('%s' % label_names[0],
                 xy=(0, 0), xycoords='data', xytext=(+0, -10), textcoords='offset points',
                 color='white')

    cnt = 1
    for i in range(1, rows):  # 1st row is black = background
        for j in range(cols):
            beg_pix = (j * persize, i * persize)  # note! (y,x)
            plt.annotate('%s' % label_names[cnt],
                         xy=beg_pix, xycoords='data', xytext=(+0, -10), textcoords='offset points',
                         color='white')
            cnt += 1

    plt.show()


if __name__ == '__main__':
    plt_label_map(rows=6, cols=8, persize=100)