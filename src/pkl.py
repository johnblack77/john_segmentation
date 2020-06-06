import time
import torch
import os
from config import args

USE_GPU = args['USE_GPU']

def save(epochnum, model, loss, optimizer, scheduler, root):

    path = os.path.join(root, time.strftime(str(epochnum) + '_%m%d_%H:%M:%S.pkl'))

    state = {
            'epoch': epochnum,
            'model': model,
            'loss' : loss,
            'optimizer': optimizer,
            'scheduler': scheduler
            }

    torch.save(state, path)



def load(path):

    if USE_GPU:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location= 'cpu')
    epoch = checkpoint['epoch']
    model = checkpoint['model']
    loss  = checkpoint['loss']

    return epoch, model, loss


def load_opti(path):

    checkpoint = torch.load(path)
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']

    return optimizer, scheduler