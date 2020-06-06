from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tfs
from PIL import Image
import numpy as np
import cv2
import torch



def transforms(img):

    img_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = img_tfs(img)

    return img



def default_loader(path):

    # return Image.open(path).convert()
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


class Dataset2d(Dataset):

    def __init__(self, txtfile_path, loader=default_loader, transform= transforms):

        path_file = open(txtfile_path, 'r')
        path_list = []

        for line in path_file:
            line = line.strip('\n')
            line = line.rstrip()
            line = line.split()
            path_list.append(line)

        self.path_list = path_list
        self.loader = loader
        self.transform = transform


    def __getitem__(self, index):

        img_path, label_path = self.path_list[index]

        img = self.loader(img_path)
        label = self.loader(label_path)

        #img = self.transform(img)
        img = torch.tensor(img).permute(2, 0, 1)
        label = torch.tensor(np.array(label, dtype=int))

        return img.float(), label


    def __len__(self):

        return len(self.path_list)

