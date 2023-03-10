# -*- coding: utf-8 -*-
"""2022dlcv_2_1_WGAN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16QI-T_HCmv5qelgMmO0fNS75cjXGZKNO
"""
"""## Download Data"""

"""## Packages"""

# Commented out IPython magic to ensure Python compatibility.
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
# print(torch.backends.mps.is_available())
# %matplotlib inline

path_to_datafile = './hw2_data/face'

# img=Image.open(os.path.join(path_to_datafile, 'p1_data/train_50/0_0.png'))
# plt.imshow(img)
# plt.show()

"""# Dataset

## Dataset
"""

class hw2_1_dataset:
    def __init__(self, filepath, transform):
        self.transform = transform
        self.file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
        self.file_list.sort()
        self.filepath = filepath
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.filepath, self.file_list[idx])
        img = Image.open(img_path)
        transformed_img = self.transform(img)
        img.close()
        return transformed_img

"""## Data Loader"""

img_transform = T.Compose([
    # T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225])
])

hw2_1_train = hw2_1_dataset(os.path.join(path_to_datafile, 'train'), img_transform)
hw2_1_test = hw2_1_dataset(os.path.join(path_to_datafile, 'val'), img_transform)
BATCH_SIZE = 128

train_loader = DataLoader(hw2_1_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(hw2_1_test, batch_size=BATCH_SIZE, shuffle=False)

"""# Model

## Generator
"""

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
        return x

"""## Discriminator"""

'''
source: https://github.com/pytorch/examples/blob/main/dcgan/main.py
        https://zhuanlan.zhihu.com/p/25071913
        https://zhuanlan.zhihu.com/p/72987027
'''
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 1).squeeze()
        return x

"""## Define Models"""

device = torch.device("cuda")
G = Generator()
D = Discriminator()
if(torch.cuda.is_available()):
    G = G.to(device)
    D = D.to(device)
else:
    print('WARNING!!!!!!!!!!!!!! MPS CAN\'T BE USED')
# criterion = nn.BCELoss()
G_optimizer = torch.optim.RMSprop(G.parameters(), lr=0.0005)
D_optimizer = torch.optim.RMSprop(D.parameters(), lr=0.0005)

"""# Training"""

'''
source: https://github.com/pytorch/examples/blob/main/dcgan/main.py
'''
# from tqdm import tqdm
import math
torch.manual_seed(5487)
EPOCH = 1000
G.train()
D.train()
n_train = len(hw2_1_train)
for epoch in range(EPOCH):
    # progress = tqdm(total = math.ceil(n_train/BATCH_SIZE))
    # total_loss_D = 0
    # total_loss_G = 0
    acc_real_D = 0
    acc_fake_D = 0
    acc_G = 0
    for batch_idx, imgs in enumerate(train_loader):
        ############################
        # (1) Update D network
        ###########################
        # train with real
        for p in D.parameters():
            p.data.clamp_(-0.1, 0.1)
        D.zero_grad()
        imgs = imgs.to(device)
        labels_D = torch.full((imgs.size()[0], ), 1, dtype=torch.float32, device=device)

        out_D = D(imgs)
        # loss_D_1 = criterion(out_D, labels_D)
        out_D.backward(labels_D)
        D_real_class = torch.sum(out_D).item()

        # train with fake
        z = torch.randn(imgs.size()[0], 100, 1, 1, device=device)
        out_G = G(z)
        labels_D.fill_(0)
        out_D = D(out_G.detach())
        # loss_D_0 = criterion(out_D, labels_D)
        out_D.backward(labels_D)
        D_optimizer.step()
        D_fake_class = torch.sum(out_D).item()

        # loss_D = loss_D_0 + loss_D_1

        # total_loss_D += loss_D.item()
        acc_real_D += D_real_class
        acc_fake_D += D_fake_class
        ############################
        # (2) Update G network
        ###########################
        if (not batch_idx % 4):
            G.zero_grad()
            labels_G = torch.full((imgs.size()[0], ), 1, dtype=torch.float32, device=device)  # fake labels are real for generator cost
            out_D = D(out_G)
            # loss_G = criterion(out_D, labels_G)
            out_D.backward(labels_G)
            G_optimizer.step()

            G_class = torch.sum(out_D).item()

            # total_loss_G += loss_G.item()
            acc_G += G_class

        # progress.update(1)
    print(epoch)
    # print("loss D:", total_loss_D/math.ceil(n_train/BATCH_SIZE))
    # print("loss G:", total_loss_G/math.ceil(n_train/BATCH_SIZE))
    print('D_real Acc:', acc_real_D/n_train)
    print('D_fake Acc:', acc_fake_D/n_train)
    print('G Acc:', acc_G/n_train)
    if(not epoch % 50):
        torch.save(D.state_dict(), f'./model_dict/big_D_{epoch}.ckpt')
        torch.save(G.state_dict(), f'./model_dict/big_G_{epoch}.ckpt')
        with torch.no_grad():
            z_eval = torch.randn(5, 100, 1, 1, device=device)
            gen_figs = G(z_eval)
            gen_figs = gen_figs.cpu()
            # fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(16, 4))
            for i in range(5):
                temp = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                        std=[1/0.229, 1/0.224, 1/0.225])(gen_figs[i])
                temp = temp.transpose(0, 2).transpose(0, 1)
                img = np.array(temp)
                im = Image.fromarray(np.uint8(img))
                im.save(f"./pic/big_{epoch}_{i}.png")