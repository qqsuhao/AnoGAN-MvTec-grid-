# -*- coding:utf8 -*-
# @TIME     : 2020/12/4 12:07
# @Author   : SuHao
# @File     : anogan.py

'''
reference:
'''


from __future__ import print_function
import os
import tqdm
import torch
from torch.utils.data import DataLoader
from models.CNN_anogan import Generator, Discriminator
# from models.upsample_anogan import Generator, Discriminator
from dataload.dataload import load_dataset
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/anogan_train", help="path to save experiments results")
parser.add_argument("--dataset", default="grid", help="mnist")
parser.add_argument('--dataroot', default=r"../../../mvtec", help='path to dataset')
parser.add_argument("--n_epoches", type=int, default=200, help="number of epoches of training")
parser.add_argument("--batchSize", type=int, default=132, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--nz", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--imageSize", type=int, default=128, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=3, help="number of image channels")
parser.add_argument("--gf_dim", type=int, default=64, help="channels of middle layers for generator")
parser.add_argument("--df_dim", type=int, default=64, help="channels of middle layers for discriminator")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
parser.add_argument("--gen_pth", default=r"../experiments/anogan_train/gen.pth", help="pretrained model of gen")
parser.add_argument("--disc_pth", default=r"../experiments/anogan_train/disc.pth", help="pretrained model of disc")
opt = parser.parse_args()
print(opt)
os.makedirs(opt.experiment, exist_ok=True)

## random seed
# opt.seed = 42
# torch.manual_seed(opt.seed)
# np.random.seed(opt.seed)

## cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## dataset
train_dataset = load_dataset(opt.dataroot, opt.dataset, opt.imageSize, trans=None, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)


## model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

## model
gen = Generator(opt.nz, opt.nc, opt.gf_dim, ksize=4).to(device)
disc = Discriminator(opt.nc, opt.df_dim).to(device)
# gen = Generator(opt.imageSize, opt.nz, opt.nc).to(device)
# disc = Discriminator(opt.imageSize, opt.nc).to(device)
gen.apply(weights_init)
disc.apply(weights_init)
if opt.gen_pth:
    gen.load_state_dict(torch.load(opt.gen_pth))
    disc.load_state_dict(torch.load(opt.disc_pth))
    print("Pretrained models have been loaded.")

## adversarial loss
gen_optimizer = optim.Adam(gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
disc_optimizer = optim.Adam(disc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
gen_criteria = nn.BCELoss()
disc_criteria = nn.BCELoss()

## record results
writer = SummaryWriter("../runs{0}".format(opt.experiment[1:]), comment=opt.experiment[1:])

## Gaussian Distribution
def gen_z_gauss(i_size, nz):
    return torch.randn(i_size, nz, 1, 1).to(device)


opt.dataSize = train_dataset.__len__()


## Training
record = 0
with tqdm.tqdm(range(opt.n_epoches)) as t:
    for e in t:
        t.set_description(f"Epoch {e+1} /{opt.n_epoches} Per epoch {train_dataset.__len__()}")
        for inputs, _, _ in train_dataloader:
            gen_epoch_loss = 0.0
            disc_epoch_loss = 0.0

            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            # inputs = inputs.view([batch_size, -1, opt.imageSize, opt.imageSize])
            label_real = torch.ones(batch_size).to(device)
            label_fake = torch.zeros(batch_size).to(device)

            # Update "D": max log(D(x)) + log(1-D(G(z))
            disc_optimizer.zero_grad()

            _, D_real = disc(inputs)
            disc_loss_real = disc_criteria(D_real, label_real)
            disc_loss_real.backward()

            noise = gen_z_gauss(batch_size, opt.nz)
            outputs = gen(noise)
            _, D_fake = disc(outputs.detach())
            disc_loss_fake = disc_criteria(D_fake, label_fake)
            disc_loss_fake.backward()

            disc_loss = (disc_loss_fake + disc_loss_real) * 0.5
            disc_optimizer.step()
            disc_epoch_loss += disc_loss.item() * batch_size

            # Update 'G' : max log(D(G(z)))
            gen_optimizer.zero_grad()
            noise = gen_z_gauss(batch_size, opt.nz)
            outputs = gen(noise)
            _, D_fake = disc(outputs)
            gen_loss = gen_criteria(D_fake, label_real)
            gen_loss.backward()
            gen_optimizer.step()
            gen_epoch_loss += gen_loss.item() * batch_size

            ## record results
            if record % opt.sample_interval == 0:
                # outputs.data = outputs.data.mul(0.5).add(0.5)
                vutils.save_image(outputs.view(-1, opt.nc, opt.imageSize, opt.imageSize),
                                  '{0}/outputs_{1}.png'.format(opt.experiment, record))
                vutils.save_image(inputs.view(-1, opt.nc, opt.imageSize, opt.imageSize),
                                  '{0}/inputs_{1}.png'.format(opt.experiment, record))
            record += 1

        ## End of epoch
        gen_epoch_loss /= opt.dataSize
        disc_epoch_loss /= opt.dataSize
        t.set_postfix(gen_epoch_loss=gen_epoch_loss, disc_epoch_loss=disc_epoch_loss)

        writer.add_scalar("gen_epoch_loss", gen_epoch_loss, e)
        writer.add_scalar("disc_epoch_loss", disc_epoch_loss, e)

        if (e+1) % 100 == 0:
        # save model parameters
            torch.save(gen.state_dict(), '{0}/gen_{1}.pth'.format(opt.experiment, e))
            torch.save(disc.state_dict(), '{0}/disc_{1}.pth'.format(opt.experiment, e))

writer.close()