from tensorboardX import SummaryWriter
from torch.autograd import Variable
import os
import torch
import time
from config import args

USE_GPU            = args['USE_GPU']
SUMMARY_PATH       = args['SUMMARY_PATH']
TrainFineTest      = args['TrainFineTest']
BATCH_SIZE         = args['BATCH_SIZE']
CHANNAL_SIZE       = args['CHANNAL_SIZE']
HIGH_SIZE          = args['HIGH_SIZE']
WIDTH_SIZE         = args['WIDTH_SIZE']




def write_train_scalar(writer, dict):

    for k in dict.keys():

        if k != 'step' and k != 'train_class_accuracy':

            writer.add_scalar(tag=os.path.join('train', k), scalar_value=dict[k], global_step=dict['step'])

        elif k == 'train_class_accuracy':

            writer.add_scalars(k, dict[k], dict['step'])





def write_val_epochlos_scalar(writer, dict):

    for k in dict.keys():

        if k != 'epoch' and k != 'val_class_accuracy':

            writer.add_scalar(tag=os.path.join('val', k), scalar_value=dict[k], global_step=dict['epoch'])

        elif k == 'val_class_accuracy':

            writer.add_scalars(k, dict[k], dict['epoch'])




def write_loss_meter_epoch_scalar(writer, dic):

    for k in dic.keys():

        if k != 'epoch':

            writer.add_scalar(os.path.join('loss_meter_epoch', k), dic[k], dic['epoch'])





def write_graph(writer, model):

    input = Variable( torch.rand(BATCH_SIZE, CHANNAL_SIZE, HIGH_SIZE, WIDTH_SIZE) )
    if USE_GPU:
        input = input.cuda()
    writer.add_graph(model,(input,))





def write_figure(writer, fig, epoch):

    writer.add_figure(tag='val_confuse', figure=fig, global_step=epoch, close=True, walltime=None)





def write_picture(writer, name, img, epoch):


    writer.add_image(name, img, epoch)



