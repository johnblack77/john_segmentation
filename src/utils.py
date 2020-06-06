import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision import transforms
from PIL import Image
from ptflops import get_model_complexity_info
#from flopth import flopth
from torchstat import stat
import config, torchsummary
from boardx import *

np.seterr(divide='ignore', invalid='ignore')

USE_GPU         = config.args['USE_GPU']
NUM_CLASS       = config.args['NUM_CLASS']
HIGH_SIZE       = config.args['HIGH_SIZE']
WIDTH_SIZE      = config.args['WIDTH_SIZE']


def confusevision(confusion):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)

    ax.set_xticklabels([''] + config.all_categories, rotation=90)
    ax.set_yticklabels([''] + config.all_categories)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.show()

    return fig



def confusevision_test(confusion, path):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)

    ax.set_xticklabels([''] + config.all_categories, rotation=90)
    ax.set_yticklabels([''] + config.all_categories)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(os.path.join(path, 'matrix.png'))
    #plt.show()



def fast_hist(label, predict, num_class):

    k = (label >= 0) & (label < num_class)
    return np.bincount(num_class * label[k].astype(int) + predict[k], minlength=num_class**2).reshape(num_class, num_class)



def get_hist(predictions, labels):

    predictions = predictions.permute(0,2,3,1)
    labels = labels.unsqueeze(dim=3)
    num_class = predictions.shape[3]
    batch_size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    if USE_GPU:
        predictions = predictions.cpu()
        labels = labels.cpu()
    predictions = predictions.detach().numpy()
    labels = labels.detach().numpy()
    for i in range(batch_size):
      hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    return hist



def print_hist_summery(hist):

    fig = confusevision(hist)

    PA = np.diag(hist).sum() / hist.sum()
    print('  PA  = %f' %PA)
    msg = str(PA) + ' '
    dic = {}
    dic['PA'] = PA

    mPA = np.nanmean(np.diag(hist) / hist.sum(axis=1))
    print(' mPA  = %f' %mPA)
    msg += str(mPA) + ' '
    dic['mPA'] = mPA

    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mIoU = np.nanmean(iu)
    print(' mIoU = %f' %mIoU)
    msg += str(mIoU) + ' '
    dic['mIoU'] = mIoU

    freq = hist.sum(axis=1) / hist.sum()
    fwIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    print('fwIoU = %f' %fwIoU)
    msg += str(fwIoU) + ' '
    dic['fwIoU'] = fwIoU
    dics = {}
    for ii in range(hist.shape[0]):

        if float(hist.sum(1)[ii]) == 0:
          acc = 0.0

        else:
          acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])

        print("    class # %2d accuracy = %f "%(ii, acc))
        msg += str(ii) + ':' + str(acc) + ' '
        name = 'class_' + str(ii)
        dics[name] = acc

    msg += '\n'
    dic['val_class_accuracy'] = dics
    return msg, dic, fig




def print_hist_summery_test(epoch, root, hist, loss):
    msg  = 'epoch      =        ' + str(epoch) + '\n'
    msg += 'test loss =         ' + str(loss) + '\n'
    PA = np.diag(hist).sum() / hist.sum()
    print('  PA  = %f' %PA)
    msg += '  PA  = ' + str(PA) + '\n'

    mPA = np.nanmean(np.diag(hist) / hist.sum(axis=1))
    print(' mPA  = %f' %mPA)
    msg += ' mPA  = ' + str(mPA) + '\n'

    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mIoU = np.nanmean(iu)
    print(' mIoU = %f' %mIoU)
    msg += ' mIoU = ' + str(mIoU) + '\n'

    freq = hist.sum(axis=1) / hist.sum()
    fwIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    print('fwIoU = %f' %fwIoU)
    msg += 'fwIoU = ' + str(fwIoU) + '\n'

    msg += '----------------------------------\n'

    for ii in range(hist.shape[0]):

        if float(hist.sum(1)[ii]) == 0:
          acc = 0.0

        else:
          acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])

        print("    class # %2d accuracy = %f "%(ii, acc))
        msg += '    class # ' + str(ii) + ' accuracy = ' + str(acc) + '\n'

    msg +='\n\n\n\n\n'
    file = open(os.path.join(root, 'result.txt'), 'a')
    file.write(msg)
    file.close()

    confusevision_test(hist, root)



def per_class_acc(predict, label):

    predict = predict.permute(0,2,3,1)
    label = label.unsqueeze(dim=3)
    size = predict.shape[0]
    num_class = predict.shape[3]
    hist = np.zeros((num_class, num_class))
    if USE_GPU:
        predict = predict.cpu()
        label = label.cpu()
    predict = predict.detach().numpy()
    label = label.detach().numpy()

    for i in range(size):
        hist += fast_hist(label[i].flatten(), predict[i].argmax(2).flatten(), num_class)

    dict = {}

    PA = np.diag(hist).sum() / hist.sum()
    print('  PA  = %f' %PA)
    msg = str(PA) + ' '
    dict['train_PA'] = PA

    mPA = np.nanmean(np.diag(hist) / hist.sum(axis=1))
    print(' mPA  = %f' %mPA)
    msg += str(mPA) + ' '
    dict['train_mPA'] = mPA

    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mIoU = np.nanmean(iu)
    print(' mIoU = %f' %mIoU)
    msg += str(mIoU) + ' '
    dict['train_mIoU'] = mIoU

    freq = hist.sum(axis=1) / hist.sum()
    fwIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    print('fwIoU = %f' %fwIoU)
    msg += str(fwIoU) + ' '
    dict['train_fwIoU'] = fwIoU

    perclass = {}
    for ii in range(num_class):

        if float(hist.sum(1)[ii]) == 0:
          acc = 0.0

        else:
          acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])

        print("    class # %2d accuracy = %f "%(ii,acc))
        msg += str(ii) + ':' + str(acc) + ' '
        name = 'train_class_' + str(ii)
        perclass[name] = acc

    dict['train_class_accuracy'] = perclass
    msg += '\n'
    return msg, dict




def writeImage(predict, path):

    if USE_GPU:
        predict =predict.cpu()
    predict = torch.argmax(predict,dim=1).detach().numpy()

    for i in range(int(np.shape(predict)[0])):

        image = predict[i]
        r = image.copy()
        g = image.copy()
        b = image.copy()
        label_colours = config.label_colours

        for l in range(0, NUM_CLASS):
            r[image==l] = label_colours[l,0]
            g[image==l] = label_colours[l,1]
            b[image==l] = label_colours[l,2]

        rgb = np.zeros((image.shape[0], image.shape[1], 3))
        rgb[:,:,0] = r/1.0
        rgb[:,:,1] = g/1.0
        rgb[:,:,2] = b/1.0

        im = Image.fromarray(np.uint8(rgb))
        im.save(path + '_' + str(i+1) + '.png')




def printparameters(model, loss, input_size= None, savepath=None, use_gpu=False):

    if use_gpu == True:
        device = 'cuda'

    else:
        device = 'cpu'

    msg = str(model) + '\n'
    print(str(model))
    msg += str(loss) + '\n\n\n\n\n'
    print(str(loss))

    bach_size = input_size[0]
    print('Batch_size:  ' + str(bach_size))
    msg += 'Batch_size:  ' + str(bach_size) + '\n'
    input_size = tuple(input_size[1:])
    msg += torchsummary.summary_with_save(model, input_size=input_size, batch_size=bach_size, device=device)
    msg += '\n\n\n'

    print('Batch_size:  ' + str(1))
    msg += 'Batch_size:  ' + str(1) + '\n'
    msg += torchsummary.summary_with_save(model, input_size=input_size, batch_size=1, device=device)
    msg += '\n\n\n\n\n'

    flops, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=True)
    print('Batch_size:  ' + str(1))
    msg += 'Batch_size: ' + str(1) + '\n'
    print("model: %s |%s" % (flops, params))
    msg += 'model-flops:      ' + str(flops) + '\n'
    msg += 'model-params:     ' + str(params) + '\n\n\n\n'

    # flo = flopth(model,in_size=[CHANNAL_SIZE, HIGH_SIZE, WIDTH_SIZE])
    # print('Batch_size:  ' + str(1))
    # msg += 'Batch_size: ' + str(1) + '\n'
    # print(flo)
    # msg += 'flops:      ' + str(flo) + '\n\n\n\n'
'''
    if savepath:
        print('Batch_size:  ' + str(1))
        msg += 'Batch_size: ' + str(1) + '\n'
        if USE_GPU:
            model = model.cpu()
        msg += stat(model, input_size)
        if USE_GPU:
            model = model.cuda()

        file = open(os.path.join(savepath, 'parameters.txt'), 'w')
        file.write(msg)
        file.close()
'''


def writeImage_train(writer, batch_label, predict, epoch):

    if USE_GPU:
        predict = predict.cpu()
        batch_label = batch_label.cpu()
    predict = torch.argmax(predict,dim=1).detach().numpy()
    batch_label = batch_label.numpy()

    for i in range(int(np.shape(predict)[0])):

        image = predict[i]
        r = image.copy()
        g = image.copy()
        b = image.copy()
        label_colours = config.label_colours

        for l in range(0, NUM_CLASS):
            r[image==l] = label_colours[l,0]
            g[image==l] = label_colours[l,1]
            b[image==l] = label_colours[l,2]

        rgb = np.zeros((image.shape[0], image.shape[1], 3))
        rgb[:,:,0] = r/1.0
        rgb[:,:,1] = g/1.0
        rgb[:,:,2] = b/1.0

        im = transforms.ToTensor()(Image.fromarray(np.uint8(rgb)))

        img = batch_label[i]
        R = img.copy()
        G = img.copy()
        B = img.copy()
        label_colours = config.label_colours

        for l in range(0, NUM_CLASS):
            R[img == l] = label_colours[l, 0]
            G[img == l] = label_colours[l, 1]
            B[img == l] = label_colours[l, 2]

        RGB = np.zeros((img.shape[0], img.shape[1], 3))
        RGB[:, :, 0] = R / 1.0
        RGB[:, :, 1] = G / 1.0
        RGB[:, :, 2] = B / 1.0

        imgl = transforms.ToTensor()(Image.fromarray(np.uint8(RGB)))


        write_picture(writer, str(epoch)+ '/predict', im, epoch)
        write_picture(writer, str(epoch)+ '/label', imgl, epoch)
