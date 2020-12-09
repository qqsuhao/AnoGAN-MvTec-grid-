# -*- coding:utf8 -*-
# @TIME     : 2020/12/4 12:22
# @Author   : SuHao
# @File     : CNN_anogan.py

'''
reference: https://github.com/seokinj/anoGAN
'''


import torch
import torch.nn as nn
from torchsummary import summary

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, gf_dim, ksize):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, gf_dim*8, ksize, 1, 0, bias=False),
            nn.BatchNorm2d(gf_dim*8),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(gf_dim*8, gf_dim*4, ksize, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim*4),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(gf_dim*4, gf_dim*2, ksize, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim*2),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(gf_dim*2, gf_dim, ksize, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(gf_dim, gf_dim>>1, ksize, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim>>1),
            nn.ReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(gf_dim>>1, c_dim, ksize, 2, 1, bias=False),
            nn.Tanh(),
        )


    def forward(self, inputs):
        outputs = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(inputs)))))
        return outputs


class Discriminator(nn.Module):
    def __init__(self, c_dim, df_dim):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(c_dim, df_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(df_dim, df_dim*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(df_dim*2, df_dim*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_dim*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(df_dim*4, df_dim*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_dim*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(df_dim*8, df_dim*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_dim*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(df_dim*16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        h4 = self.layer4(self.layer3(self.layer2(self.layer1(inputs))))
        outputs = self.layer5(h4)
        return h4, outputs.view(-1, 1).squeeze(1)        # by squeeze, get just float not float Tenosor



def print_net():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(100, 3, 64, 4).to(device)
    D = Discriminator(3, 64).to(device)
    summary(G, (100, 1, 1))
    summary(D, (3, 128, 128))


print_net()