from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.nn.functional as F
from config import args, loss_weight


USE_GPU         = args['USE_GPU']
NUM_CLASS       = args['NUM_CLASS']



loss_weight     = Variable(torch.from_numpy(loss_weight),requires_grad=False).float()
if USE_GPU:
    loss_weight = loss_weight.cuda()


class CrossEntropyLosswithOneHot(nn.Module):

    def __init__(self):

        super(CrossEntropyLosswithOneHot, self).__init__()

        self.name = 'CrossEntropyLosswithOneHot'
        self.previous_loss = 1e100
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, input, batch_label):

        input = self.logsoftmax(input).permute(0, 2, 3, 1).reshape(-1, NUM_CLASS)
        batch_label = F.one_hot(batch_label, num_classes=NUM_CLASS).view(-1, NUM_CLASS)
        loss = torch.mean(-torch.sum(torch.mul(torch.mul(batch_label, input), loss_weight), dim=1))

        return loss


    def fix_preloss(self, pl):

        self.previous_loss = pl



class CrossEntropyLoss(nn.Module):

    def __init__(self):

        super(CrossEntropyLoss, self).__init__()
        self.name = 'CrossEntropyLoss'
        self.previous_loss = 1e100
        self.crossentropyloss = nn.CrossEntropyLoss(loss_weight)


    def forward(self, input, batch_label):

        batch_label = batch_label.view(-1).long()
        input = input.permute(0, 2, 3, 1).reshape(-1, NUM_CLASS).float()
        loss = self.crossentropyloss(input, batch_label)

        return loss


    def fix_preloss(self, pl):

        self.previous_loss = pl


class Loss(nn.Module):

    def __init__(self):

        super(Loss, self).__init__()
        self.name = 'Loss'
        self.previous_loss = 1e100

        self.nllloss = nn.NLLLoss(loss_weight)


    def forward(self, input, batch_label):

        out = F.log_softmax(input, dim=1)
        loss = self.nllloss(out,batch_label)

        return loss


    def fix_preloss(self, pl):

        self.previous_loss = pl