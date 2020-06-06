import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from PIL import Image
from matplotlib import pyplot as plt


def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

x = Image.open('./dataset/......jpg')
x = np.array(x)
plt.imshow(x)
print(x.shape)


x = torch.from_numpy(x.astype('float32')).permute(2, 0, 1).unsqueeze(0)
# 定义转置卷积
conv_trans = nn.ConvTranspose2d(3, 3, 4, 2, 1)
# 将其定义为 bilinear kernel
conv_trans.weight.data = bilinear_kernel(3, 3, 4)

y = conv_trans(Variable(x)).data.squeeze().permute(1, 2, 0).numpy()
plt.imshow(y.astype('uint8'))
plt.show()
print(y.shape)



