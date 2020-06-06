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




def train(load_model_path=None):

    if load_model_path:

        print('LOAD_MODEL_PATH:  ' + load_model_path)
        epoch, model, criterion = pkl.load(load_model_path)

        if USE_GPU:
            model.cuda()
            criterion.cuda()

        #optimizer, scheduler = pkl.load_opti(load_model_path)
        lr = optimizer.param_groups[0]['lr']


    else:

        epoch =0
        model = Model(NUM_CLASS)

        if  ONE_HOTE:
            criterion = CrossEntropyLosswithOneHot()
        else:
            criterion = CrossEntropyLoss()
        criterion = Loss()

        if USE_GPU:
            model.cuda()
            criterion.cuda()

        lr = LR
        optimizer, scheduler = init_optimizer(model, name=OPTIMZIER, lr=lr, weight_decay=weight_decay, factor=lr_decay)





    train_data = Dataset2d(txtfile_path=os.path.join(DATA_PATH, 'train.txt'))
    val_data = Dataset2d(txtfile_path=os.path.join(DATA_PATH, 'val.txt'))

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=NON_BLOCKING, pin_memory=PIP_MEMORY, shuffle=True)  #     drop_last=True
    val_loader  = DataLoader(dataset=val_data,  batch_size=BATCH_SIZE, num_workers=NON_BLOCKING, pin_memory=PIP_MEMORY, shuffle=True)


    strtime = time.strftime('%m%d_%H:%M:%S')
    train_log_path = os.path.join(TRAIN_LOG_PATH, strtime)
    os.makedirs(train_log_path)
    fig_path = os.path.join(train_log_path, 'confuse')
    os.makedirs(fig_path)
    logspath = os.path.join(train_log_path, 'trainslog.txt')
    logelospath = os.path.join(train_log_path , 'traineloslog.txt')
    logvalpath = os.path.join(train_log_path, 'vallog.txt')
    stitle = 'DateTime ' + 'step ' + 'epoch ' + 'batch_in_epoch ' + 'train_step_loss ' + 'examples/sec ' + 'sec/batch ' + 'PA ' + 'mPA ' + 'mIoU ' +'fwIoU ' + 'class_accuracy' + '\n'
    file = open(logspath, 'a')
    file.write(stitle)
    file.close()
    etitle = 'epoch ' + 'train_loss_meter_epoch' + '\n'
    file = open(logelospath, 'a')
    file.write(etitle)
    file.close()
    vtitle = 'epoch ' + 'val_loss_meter_epoch ' + 'PA ' + 'mPA ' + 'mIoU ' +'fwIoU ' + 'class_accuracy' + '\n'
    file = open(logvalpath, 'a')
    file.write(vtitle)
    file.close()

    save_model_path = os.path.join(SAVE_MODEL_PATH, strtime)
    os.makedirs(save_model_path)

    summary_path = os.path.join(SUMMARY_PATH, time.strftime('%m%d_%H:%M:%S'))
    writer = SummaryWriter(logdir=summary_path)


    #write_graph(writer, model)


    print('BATCH_SIZE：      ' + str(BATCH_SIZE) )
    print('EPOCH:            ' + str(epoch))
    print('MAX_EPOCH:        ' + str(MAX_EPOCH))
    print('USE_GPU:          ' + str(USE_GPU))
    print('ONE_HOTE:         ' + str(ONE_HOTE))
    print('LOSS:             ' + criterion.name)
    print('DATA_PATH:        ' + DATA_PATH)
    print('SAVE_MODEL_PATH:  ' + str(save_model_path))
    print('TRAIN_LOG_PATH:   ' + str(train_log_path))
    print('SUMMARY_PATH:     ' + str(summary_path))
    print('NUM_WORKER:       ' + str(NUM_WORKER))
    print('OPTIMZIER:        ' + OPTIMZIER)
    print('optimizer:      \n' + str(optimizer))
    print('lr:               ' + str(lr))
    print('lr_decay:         ' + str(lr_decay))
    print('weight_decay:     ' + str(weight_decay))
    print('PRINT_FREQ:       ' + str(PRINT_FREQ))
    print('CHANNAL_SIZE:     ' + str(CHANNAL_SIZE))
    print('HIGH_SIZE:        ' + str(HIGH_SIZE))
    print('WIDTH_SIZE:       ' + str(WIDTH_SIZE))
    print('NUM_CLASS:        ' + str(NUM_CLASS))
    print('Train_loader_len: '+str(len(train_loader)))
    print('  Val_loader_len: '+str(len(val_loader)))
    #print(str(model))
    #print(str(criterion))


    train_loss_meter = meter.AverageValueMeter()

    for epoch_ in range(epoch+1, MAX_EPOCH+1, 1):

        train_loss_meter.reset()

        model.train()
        criterion.train()
        smsg = ''
        rand = random.randint(0, len(train_loader)-1)

        for ii, (batch_input, batch_label) in enumerate(train_loader):

            start_time = time.time()


            if USE_GPU:

                batch_input = batch_input.cuda(non_blocking=NON_BLOCKING)
                batch_label = batch_label.cuda(non_blocking=NON_BLOCKING)

            batch_input, batch_label = Variable(batch_input, requires_grad=False), Variable(batch_label, requires_grad=False)

            predict = model(batch_input)
            loss = criterion(predict, batch_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_meter.add(loss.item())

            duration = time.time() - start_time


            if ((ii + 1) % PRINT_FREQ) == 0:

                format_str = ('%s: step %d, epoch %d, batch_in_epoch %d, train_step_loss = %.5f (%.5f examples/sec; %.3f  sec/batch)')
                print(format_str % (datetime.datetime.now(), (epoch_-1)*len(train_loader)+ii+1, epoch_, ii+1, loss, BATCH_SIZE / duration, float(duration)))
                smsg += (time.strftime('%Y%m%d_%H:%M:%S') + ' ' + str((epoch_-1)*len(train_loader)+ii+1) + ' ' + str(epoch_) + ' ' + str(ii+1) + ' ' + str(loss.item()) + ' ' + str(BATCH_SIZE / duration) + ' ' + str(float(duration))) + ' '
                msg , dic =  per_class_acc(predict, batch_label)
                smsg += msg

                train_summary = {}
                train_summary['step'] = (epoch_-1)*len(train_loader)+ii+1
                train_summary['train_step_loss'] = loss.item()
                train_summary.update(dic)
                write_train_scalar(writer, train_summary)


            if ii == rand:

                writeImage_train(writer, batch_label, predict, epoch_)



        file = open(logspath, 'a')
        file.write(smsg)
        file.close()

        print('train_loss_meter_epoch:    ' + str(train_loss_meter.value()[0]))
        file = open(logelospath, 'a')
        file.write(str(epoch_) + ' ' + str(train_loss_meter.value()[0]) + '\n')
        file.close()

        vlos, vmsg, val_summary,fig = val(model, criterion, val_loader)

        val_summary['epoch'] = epoch_
        los_epoch = {'epoch':epoch_}
        los_epoch['train_loss_meter_epoch'] = train_loss_meter.value()[0]
        los_epoch['val_loss_meter_epoch'] = vlos


        write_val_epochlos_scalar(writer, val_summary)
        write_loss_meter_epoch_scalar(writer, los_epoch)
        write_figure(writer, fig, epoch_)


        fig_save_path = os.path.join(fig_path, str(epoch_)+'.png')
        fig.savefig(fig_save_path)

        file = open(logvalpath, 'a')
        file.write(str(epoch_) + ' ' + vmsg)
        file.close()

        scheduler.step(vlos)
        # if epoch_ == 1:
        #     criterion.fix_preloss(vlos)
        # if vlos > criterion.previous_loss and epoch_>5:
        #     lr = lr * lr_decay
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # criterion.fix_preloss(train_loss_meter.value()[0])


        pkl.save(epoch_, model, criterion, optimizer, scheduler, save_model_path)

    writer.close()




def val(model, criterion, dataloader):

    with torch.no_grad():

        print("start validating-------------------------------")

        model.eval()
        criterion.eval()

        val_loss_meter = meter.AverageValueMeter()
        val_loss_meter.reset()
        hist = np.zeros((NUM_CLASS, NUM_CLASS))

        for ii, (batch_input, batch_label) in enumerate(dataloader):

            if USE_GPU:

                batch_input = batch_input.cuda(non_blocking=NON_BLOCKING)
                batch_label = batch_label.cuda(non_blocking=NON_BLOCKING)

            batch_input, batch_label = Variable(batch_input, requires_grad=False), Variable(batch_label, requires_grad=False)

            predict = model(batch_input)
            loss = criterion(predict, batch_label)

            val_loss_meter.add(loss.item())
            hist += get_hist(predict, batch_label)


        print('val_loss_meter_epoch:    ' + str(val_loss_meter.value()[0]))
        msg = str(val_loss_meter.value()[0]) + ' '
        msgs, val_summary, fig = print_hist_summery(hist)
        msg +=msgs

        print("  end validating-------------------------------")

        model.train()
        criterion.train()

        return val_loss_meter.value()[0], msg, val_summary, fig



def test():

    epoch, model, criterion = pkl.load(LOAD_MODEL_PATH)

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
    msg += 'TEST_SAVE_PATH:   ' + str(TEST_SAVE_PATH) + '\n'
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

    savepath = os.path.join(TEST_SAVE_PATH, time.strftime('%m%d_%H:%M:%S'))
    os.mkdir(path=savepath)
    resultimgpath = os.path.join(savepath, 'rgb_label')
    os.mkdir(path=resultimgpath)

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

            writeImage(predict, os.path.join(resultimgpath, str(ii+1)))

            test_loss_meter.add(loss.item())
            hist += get_hist(predict, batch_label)


        print('test_loss_meter_epoch:    ' + str(test_loss_meter.value()[0]))

        print_hist_summery_test(epoch, savepath, hist, test_loss_meter.value()[0])


if __name__ == '__main__':
    if TrainFineTest == 'train':

        print('Training...')
        train()

    elif TrainFineTest == 'fine':

        print('Fine tuning...')
        train(LOAD_MODEL_PATH)

    elif TrainFineTest == 'test':

        print('Testing...')
        test()

    else:

        print('TrainFineTest: err')




