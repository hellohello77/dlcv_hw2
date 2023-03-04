import torch
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("img_dir")
args = parser.parse_args()

'''
source: https://github.com/pytorch/examples/blob/main/dcgan/main.py
'''
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, 1024, kernel_size = 4, stride = 1, padding = 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(1024, 512, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(128, 3, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        x = self.main(x)
        # x = (x+1)/2
        return x

device = torch.device("cpu")
G = Generator()
G.load_state_dict(torch.load('DCGAN_G152.ckpt', map_location=torch.device('cpu')))
with torch.no_grad():
    torch.manual_seed(5487)
    z_eval = torch.randn(1000, 100, 1, 1, device=device)
    G.eval()
    gen_figs = G(z_eval)
    gen_figs = gen_figs.cpu()
    for i in range(1000):
        temp = T.Normalize(mean=[-1, -1, -1],
                std=[2, 2, 2])(gen_figs[i])
        
        temp = temp.transpose(0, 2).transpose(0, 1)
        # print(temp.size())
        img = np.array(temp)
        im = Image.fromarray((img * 255).astype(np.uint8))
        im.save(os.path.join(args.img_dir, f'{str(i).zfill(4)}.png'))