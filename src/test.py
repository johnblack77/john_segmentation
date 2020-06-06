from dataset import *
from models import *
from loss import *
from utils import *
from optimizer import *
from boardx import *
from config import args
import pkl
import torch
from torchnet import meter
import random
import datetime
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



DATA_PATH       = args['DATA_PATH']
BATCH_SIZE      = args['BATCH_SIZE']
NUM_WORKER      = args['NUM_WORKER']
USE_GPU         = args['USE_GPU']
PIP_MEMORY      =       USE_GPU
NON_BLOCKING    =       PIP_MEMORY
LOAD_MODEL_PATH = args['LOAD_MODEL_PATH']
MAX_EPOCH       = args['MAX_EPOCH']
PRINT_FREQ      = args['PRINT_FREQ']
NUM_CLASS       = args['NUM_CLASS']
weight_decay    = args['weight_decay']
ONE_HOTE        = args['ONE_HOTE']
TrainFineTest   = args['TrainFineTest']
SAVE_MODEL_PATH = args['SAVE_MODEL_PATH']
LR              = args['lr']
lr_decay        = args['lr_decay']
OPTIMZIER       = args['OPTIMZIER']
TEST_SAVE_PATH  = args['TEST_SAVE_PATH']
TEST_DATA_PATN  = args['TEST_DATA_PATN']
WIDTH_SIZE      = args['WIDTH_SIZE']
HIGH_SIZE       = args['HIGH_SIZE']
CHANNAL_SIZE    = args['CHANNAL_SIZE']
SUMMARY_PATH    = args['SUMMARY_PATH']
TRAIN_LOG_PATH  = args['TRAIN_LOG_PATH']

BATCH_TEST_SAVE_PATH = args['BATCH_TEST_SAVE_PATH']
pkldirpath = '/home/jhon/Documents/john_segmentation/data/modellog/0519_13:43:23/'



def batchtest():

    savepath = os.path.join(BATCH_TEST_SAVE_PATH, time.strftime('%m%d_%H:%M:%S'))
    os.mkdir(path=savepath)
    resultimgpath = os.path.join(savepath, 'rgb_label')
    os.mkdir(path=resultimgpath)

    for root, dirs, files in os.walk(pkldirpath):
        max = [0,0]
        for file in files:
            load_model_path = os.path.join(root,file)

            epoch, model, criterion = pkl.load(load_model_path)

            msg =  'BATCH_SIZE:       ' + str(BATCH_SIZE) + '\n'
            msg += 'EPOCH:            ' + str(epoch) + '\n'
            msg += 'USE_GPU:          ' + str(USE_GPU) + '\n'
            msg += 'LOSS:             ' + criterion.name + '\n'
            msg += 'NUM_WORKER:       ' + str(NUM_WORKER) + '\n'
            msg += 'CHANNAL_SIZE:     ' + str(CHANNAL_SIZE) + '\n'
            msg += 'HIGH_SIZE:        ' + str(HIGH_SIZE) + '\n'
            msg += 'WIDTH_SIZE:       ' + str(WIDTH_SIZE) + '\n'
            msg += 'NUM_CLASS:        ' + str(NUM_CLASS) + '\n'
            msg += 'TEST_DATA_PATN:   ' + TEST_DATA_PATN + '\n'
            msg += 'TEST_SAVE_PATH:   ' + str(BATCH_TEST_SAVE_PATH) + '\n'
            msg += 'LOAD_MODEL_PATH:  ' + str(LOAD_MODEL_PATH) + '\n'
            print(msg,end='')


            if USE_GPU:
                model.cuda()
                criterion.cuda()

            model.eval()
            criterion.eval()

            test_data = Dataset2d(txtfile_path=TEST_DATA_PATN)
            test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=NON_BLOCKING, pin_memory=PIP_MEMORY)
            print( 'Test_loader_len: ' + str(len(test_loader)))
            msg += 'Test_loader_len: ' + str(len(test_loader))

            test_loss_meter = meter.AverageValueMeter()
            test_loss_meter.reset()
            hist = np.zeros((NUM_CLASS, NUM_CLASS))



            file = open(os.path.join(savepath, 'hype_parameters.txt'), 'w')
            file.write(msg)
            file.close()

            #printparameters(model=model, loss=criterion, input_size=[BATCH_SIZE,CHANNAL_SIZE,HIGH_SIZE,WIDTH_SIZE], savepath =savepath, use_gpu=USE_GPU)

            with torch.no_grad():

                for ii, (batch_input, batch_label) in enumerate(test_loader):

                    if USE_GPU:
                        model = model.cuda()
                        criterion = criterion.cuda()
                        batch_input = batch_input.cuda(non_blocking=NON_BLOCKING)
                        batch_label = batch_label.cuda(non_blocking=NON_BLOCKING)

                    batch_input, batch_label = Variable(batch_input, requires_grad=False), Variable(batch_label, requires_grad=False)

                    predict = model(batch_input)
                    loss = criterion(predict, batch_label)

                    #writeImage(predict, os.path.join(resultimgpath, str(ii+1)))

                    test_loss_meter.add(loss.item())
                    hist += get_hist(predict, batch_label)


                print('test_loss_meter_epoch:    ' + str(test_loss_meter.value()[0]))

                print_hist_summery_test(epoch, savepath, hist, test_loss_meter.value()[0])

                if


if __name__ == '__main__':

    batchtest()
