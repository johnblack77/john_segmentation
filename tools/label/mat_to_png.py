'''

stage 1
seg.mat to seg1.png
seg1.png only include wall:1

'''

'''
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

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

c1 = 1

for root, dirs, files in os.walk(str1):
    if 'seg.mat' in files:
        dir = ''
        imageroot = root + dir + '/' + 'image/'
        segroot = root + dir + '/' + 'seg.mat'
        segpng = root + dir + '/' + 'seg1.png'

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
            segimg = np.zeros((h, w), dtype=int)

        segimg = gene('wall', stack, c1, segimg, data)
        # segimg = gene('hall_wall', stack, c2, segimg, data)
        # segimg = gene('back_room_wall', stack, c3, segimg, data)

        cv2.imwrite(segpng,segimg)

        c += 1
        print(c)
'''




'''

stage 2
seg.png add seg.png
seg1.png only include wall:1+2+...

2   floor       ['floor', 'panalled_floor', 'flloor', 'dark_floor']
3   cabinet     
4   bed
5   chair
6   sofa        ['sofa','sofas']
7   table
8   door
9   window
10  bookshelf   ['bookshelf','bookshelve']
11  picture     ['picture','pictur']
12  counter
13  blinds      ['blinds','blind']
14  desk        ['desk','desk_top','wall_desk','school_desk','desk_protector','desk_blotter']
15  shelves     ['shelves','shelf','shelving_division_panel','shelf\'','mail shelf','paper_shelf','shelf frame','toy shelf']
+10 bookshelf   ['book_casr','book_case','bookcase','book_self','book_sextion','short_bookcase','tall_bookcase','spiral_notebooks','book_cart']
16  curtain     ['curtain','curatin','shower_curtain','door curtain','shower curtain','curtian','bed_curtain']
17  dresser
18  pillow      ['pillow','piloow','throw_pillow','pillw']
19  mirror      ['mirror','mirro','mirror_-_center_panel','miror']
20  floormat    ['floormat','mat','floor_mat','floor_mats','floor mat','shower_mat','bath_mat','kitchen_mat','welcome_mat','mats','bathroom_mat','bath_mat','doormat']
21  clothes     ['clothes','folded_cloth','pant_cloths','hanging_clothing','cloth','drying_clothe','cloth_basckets','shitrt','shirts_with_stand']
22  ceiling     ['ceiling','ceililg','cealing','ceilin','celeing']
23  books       ['books','booka','boks','book\'','notebook','book','spiral_notebooks']
24  fridge      ['fridge','refrigrator','refrigator','frige','fredze','fridje','beavarage_refrigirator']

'''


import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
import time




# 修改

c1 =
segclas = []



str1 = '/Users/john/PycharmProjects/trail/label/data/SUNRGBD/'
count = 0
start = time.time()
def gene(name, stack, color, segimg, data):
    if name in stack:
        seg1 = data['seglabel']
        h, w = np.shape(seg1)
        for ind in range(len(stack)):
            if stack[ind] == name:

                global count
                count += 1

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
for root, dirs, files in os.walk(str1):
    if 'seg.mat' in files:

        imageroot = root +'/image/'
        segroot = root + '/seg.mat'
        segpng = root + '/seg1.png'

        segimg = cv2.imread(segpng,cv2.IMREAD_UNCHANGED)

        data = sio.loadmat(segroot)
        label = data['names']
        label = np.array(label)
        label = label.reshape(1, -1)
        le = len(label[0])
        stack = []
        for i in range(le):
            tmp = str(label[0][i])[2:-2]
            stack.append(tmp)


        for sc in segclas:
            segimg = gene(sc, stack, c1, segimg, data)


        cv2.imwrite(segpng,segimg)

        c += 1
        print(c)


print('label:'+str(c1))
print('num_label:' + str(count))
print(segclas)
end = time.time()
end = end - start
print('time:' + str(end))