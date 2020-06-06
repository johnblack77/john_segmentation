import torch
import torch.nn as nn
import numpy as np


## build model
class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        layer = self.fc1(inputs)
        layer = self.relu(layer)
        layer = self.fc2(layer)
        return layer


## analoy inputs and labels
inputs = np.random.normal(size=(8, 100))
inputs = torch.tensor(inputs).float()
labels = np.ones((8, 1))
labels = torch.tensor(labels).float()

## update the weights and bias with L2 weight decay
n = net()
weight_p, bias_p = [], []
for name, p in n.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]

'''
l1_regularization, l2_regularization = torch.tensor([0],dtype =torch.float32), torch.tensor([0],dtype=torch.float32) #定义L1及L2正则化损失
    #注意 此处for循环 当上面定义了weight_decay时候，应注释掉
    for param in model.parameters():
        l1_regularization += torch.norm(param, 1) #L1正则化
        l2_regularization += torch.norm(param, 2) #L2 正则化
    #
    #prin(loss.item())
    #loss = cross_loss + l1_regularization #L1 正则化
    loss = cross_loss + l2_regularization #L2 正则化

'''


criterion = nn.MSELoss()
logit = n(inputs)
loss = criterion(input=logit, target=labels)
opt = torch.optim.SGD([{'params': weight_p, 'weight_decay': 1e-5},
                       {'params': bias_p, 'weight_decay': 0}],
                      lr=1e-2,
                      momentum=0.9)

## update
opt.zero_grad()
loss.backward()
opt.step()






'''

很多时候如果对bb进行L2正则化将会导致严重的欠拟合

'''





'''

可视化激活与权重

'''

