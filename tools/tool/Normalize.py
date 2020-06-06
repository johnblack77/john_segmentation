from torchvision import transforms as tfs
import torch


def img_transforms(img, label):

    img_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = img_tfs(img)
    label = torch.from_numpy(label)
    return img, label


