# -*- coding: utf-8 -*-
"""2022dlcv_2_1_DCGAN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XAfUT_ndQneijCOAK5gsm-f2Ka1Ody_z

# Preparation

## Check GPU Type
"""

!nvidia-smi

from google.colab import drive
drive.mount('/content/drive')

"""## Download Data"""

!gdown --id 1YxkObGDlqZM0-9Zq-QMjk7q1vND4UJl3 --output "./hw2_data.zip"
!unzip -q "./hw2_data.zip" -d "./"
!rm hw2_data.zip

!pip3 install face-recognition==1.3.0

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

path_to_datafile = '/content/hw2_data/face'

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
    T.RandomHorizontalFlip(p = 0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
               std=[0.5, 0.5, 0.5])
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
        # x = (x+1)/2
        return x

"""## Discriminator"""

'''
source: https://github.com/pytorch/examples/blob/main/dcgan/main.py
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
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 1).squeeze()
        return x

"""## Define Models"""

ck = 111
device = torch.device("cuda")
G = Generator()
D = Discriminator()
G.load_state_dict(torch.load(f'/content/drive/MyDrive/hw2_models/DCGAN_G{ck}.ckpt'))
D.load_state_dict(torch.load(f'/content/drive/MyDrive/hw2_models/DCGAN_D{ck}.ckpt'))
if(torch.cuda.is_available()):
    G = G.to(device)
    D = D.to(device)
else:
    print('WARNING!!!!!!!!!!!!!! MPS CAN\'T BE USED')
criterion = nn.BCELoss()
G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

"""# Tools"""

import face_recognition
import os
from tqdm.notebook import tqdm

def face_recog(image_dir):
    image_ids = os.listdir(image_dir)
    total_faces = len(image_ids)
    num_faces = 0
    print("Start face recognition...")
    for image_id in tqdm(image_ids):
        image_path = os.path.join(image_dir, image_id)
        try: # Prevent unexpected file
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image, model="HOG")
            if len(face_locations) == 1:
                num_faces += 1
        except:
            total_faces -= 1
    acc = (num_faces / total_faces) * 100
    return acc

"""# Training"""

'''
source: https://github.com/pytorch/examples/blob/main/dcgan/main.py
'''
from tqdm.notebook import tqdm
import math
torch.manual_seed(5487)
EPOCH = 1000
os.makedirs('gen_img', exist_ok = True)
G.train()
D.train()
n_train = len(hw2_1_train)
# epoch ~ 120
for epoch in range(EPOCH):
    progress = tqdm(total = math.ceil(n_train/BATCH_SIZE))
    total_loss_D = 0
    total_loss_G = 0
    acc_real_D = 0
    acc_fake_D = 0
    acc_G = 0
    for batch_idx, imgs in enumerate(train_loader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        D.zero_grad()
        imgs = imgs.to(device)
        labels_D = torch.full((imgs.size()[0], ), 1, dtype=torch.float32, device=device)

        out_D = D(imgs)
        loss_D_1 = criterion(out_D, labels_D)
        loss_D_1.backward()
        D_real_class = torch.sum(out_D).item()

        # train with fake
        z = torch.randn(imgs.size()[0], 100, 1, 1, device=device)
        out_G = G(z)
        labels_D.fill_(0)
        out_D = D(out_G.detach())
        loss_D_0 = criterion(out_D, labels_D)
        loss_D_0.backward()
        D_optimizer.step()
        D_fake_class = torch.sum(out_D).item()

        loss_D = loss_D_0 + loss_D_1

        total_loss_D += loss_D.item()
        acc_real_D += D_real_class
        acc_fake_D += D_fake_class
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        G.zero_grad()
        labels_G = torch.full((imgs.size()[0], ), 1, dtype=torch.float32, device=device)  # fake labels are real for generator cost
        out_D = D(out_G)
        loss_G = criterion(out_D, labels_G)
        loss_G.backward()
        G_optimizer.step()

        G_class = torch.sum(out_D).item()

        total_loss_G += loss_G.item()
        acc_G += G_class

        progress.update(1)
    print(epoch+111+1)
    print("loss D:", total_loss_D/math.ceil(n_train/BATCH_SIZE))
    print("loss G:", total_loss_G/math.ceil(n_train/BATCH_SIZE))
    print('D_real Acc:', acc_real_D/n_train)
    print('D_fake Acc:', acc_fake_D/n_train)
    print('G Acc:', acc_G/n_train)
    # 101 30.267 112 28.38
    if(not epoch % 10):
        with torch.no_grad():
            torch.save(D.state_dict(), f'/content/drive/MyDrive/hw2_models/DCGAN_D{epoch+1+111}.ckpt')
            torch.save(G.state_dict(), f'/content/drive/MyDrive/hw2_models/DCGAN_G{epoch+1+111}.ckpt')
            z_eval = torch.randn(1000, 100, 1, 1, device=device)
            G.eval()
            gen_figs = G(z_eval)
            gen_figs = gen_figs.cpu()
            fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(16, 4))
            for i in range(1000):
                temp = T.Normalize(mean=[-1, -1, -1],
                        std=[2, 2, 2])(gen_figs[i])
                
                temp = temp.transpose(0, 2).transpose(0, 1)
                # print(temp.size())
                img = np.array(temp)
                if(i<5):
                  axes[i].imshow(img)
                  axes[i].set_axis_off()
                im = Image.fromarray((img * 255).astype(np.uint8))
                im.save(f'gen_img/{str(i).zfill(4)}.png')
            gen_acc = face_recog('gen_img')
            print('acc: ', gen_acc)
            plt.tight_layout()
            plt.show()
            G.train()