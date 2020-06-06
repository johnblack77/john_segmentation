from torch import nn
import torch
import numpy as np
from config import args

CHANNAL_SIZE = args['CHANNAL_SIZE']
HIGH_SIZE    = args['HIGH_SIZE']
WIDTH_SIZE   = args['WIDTH_SIZE']


def bilinear_kernel(in_channels, out_channels, kernel_size):

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


def initialize_weights(*models):

    for model in models:

        for module in model.modules():

            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):

                nn.init.kaiming_normal_(module.weight)

                if module.bias is not None:

                    module.bias.data.zero_()

            elif isinstance(module, nn.BatchNorm2d):

                module.weight.data.fill_(1)
                module.bias.data.zero_()

            elif isinstance(module, nn.ConvTranspose2d):

                ConvT_shape = module.weight.shape
                in_size = ConvT_shape[0]
                out_size = ConvT_shape[1]
                kernel_size = ConvT_shape[2]
                module.weight.data = bilinear_kernel(in_size, out_size, kernel_size)
                # module.weight.requires_grad=False


class Model(nn.Module):

    def __init__(self, NUM_CLASS):

        super(Model, self).__init__()


        self.lrn = torch.nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0)


        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gpool = nn.AdaptiveAvgPool2d((HIGH_SIZE, WIDTH_SIZE))
        self.gconv = nn.Sequential(nn.Conv2d(in_channels=512,  out_channels=8,  kernel_size=1, stride=1, padding=0, padding_mode='replicate'), nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        self.fconv = nn.Sequential(nn.Conv2d(in_channels=72,   out_channels=64,  kernel_size=3, stride=1, padding=1, padding_mode='replicate'),nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.upsamping1 = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, groups=1, bias=False, dilation=1, padding_mode='zeros'))
        self.upsamping2 = nn.Sequential(nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1, groups=1, bias=False, dilation=1, padding_mode='zeros'))
        self.upsamping3 = nn.Sequential(nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1, groups=1, bias=False, dilation=1, padding_mode='zeros'))


        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=CHANNAL_SIZE, out_channels=64,  kernel_size=3, stride=1, padding=1, padding_mode='replicate'), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,  out_channels=64,  kernel_size=3, stride=1, padding=1, padding_mode='replicate'), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        self.encoder1_c = nn.Sequential(nn.Conv2d(in_channels=64,  out_channels=8,  kernel_size=1, stride=1, padding=0, padding_mode='replicate'), nn.BatchNorm2d(8), nn.ReLU(inplace=True))


        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='replicate'), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )

        self.encoder2_c = nn.Sequential(nn.Conv2d(in_channels=128,  out_channels=16,  kernel_size=1, stride=1, padding=0, padding_mode='replicate'), nn.BatchNorm2d(16), nn.ReLU(inplace=True))


        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )

        self.encoder3_c = nn.Sequential(nn.Conv2d(in_channels=256,  out_channels=32,  kernel_size=1, stride=1, padding=0, padding_mode='replicate'), nn.BatchNorm2d(32), nn.ReLU(inplace=True))


        self.encoder4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, padding_mode='replicate'), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, padding_mode='replicate'), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, padding_mode='replicate'), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2, padding_mode='replicate'), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )

        self.encoder5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='replicate'), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='replicate'), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='replicate'), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='replicate'), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )


        self.decoder3 = nn.Sequential(
            nn.Conv2d(in_channels=544, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode='replicate'), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(in_channels=272, out_channels=128,  kernel_size=3, stride=1, padding=1, padding_mode='replicate'), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128 , kernel_size=3, stride=1, padding=1, padding_mode='replicate'), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_channels=136, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        self.class_out = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=NUM_CLASS, kernel_size=1, stride=1, padding=0, padding_mode='replicate')
        )


        initialize_weights(self.gconv,self.fconv,self.upsamping1,self.upsamping2,self.upsamping3,self.encoder1,self.encoder1_c,self.encoder2,self.encoder2_c,self.encoder3,
            self.encoder3_c,self.encoder4,self.encoder5,self.decoder3,self.decoder2,self.decoder1,self.class_out)



    def forward(self, input):
        feat = self.lrn(input)
        feat = self.encoder1(feat)
        fc1  = self.encoder1_c(feat)
        feat = self.pool(feat)
        feat = self.encoder2(feat)
        fc2  = self.encoder2_c(feat)
        feat = self.pool(feat)
        feat = self.encoder3(feat)
        fc3  = self.encoder3_c(feat)
        feat = self.pool(feat)

        feata = self.encoder4(feat)
        feat  = self.encoder5(feata)
        feat  = torch.cat((feata,feat),dim=1)
        gfeat = self.gpool(feat)
        gfeat = self.gconv(gfeat)

        feat = self.upsamping3(feat)
        feat = torch.cat((feat,fc3), dim=1)
        feat = self.decoder3(feat)
        feat = self.upsamping2(feat)
        feat = torch.cat((feat,fc2),dim=1)
        feat = self.decoder2(feat)
        feat = self.upsamping1(feat)
        feat = torch.cat((feat,fc1),dim=1)
        feat = self.decoder1(feat)

        feat = torch.cat((feat,gfeat),dim=1)
        feat = self.fconv(feat)

        feat = self.class_out(feat)

        return feat


