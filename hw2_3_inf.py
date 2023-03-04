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
import csv
import argparse
from torch.autograd import Function

parser = argparse.ArgumentParser()
parser.add_argument("img_dir")
parser.add_argument("dst_file")
args = parser.parse_args()

print(args.img_dir.split('/')[-2])
if 'usps' in args.img_dir.split('/'):
    target_domain = 'usps'
    channel = 1
    model_dict = './DANN_test11_usps_81.ckpt'
elif 'svhn' in args.img_dir.split('/'):
    target_domain = 'svhn'
    channel = 3
    model_dict = './DANN_39_504.ckpt'

class hw2_3_dataset_val:
    def __init__(self, filepath, transform):
        self.transform = transform
        self.filepath = filepath
        self.file_list = [file for file in os.listdir(filepath)]
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.filepath, self.file_list[idx])
        if target_domain == 'usps':
            img = Image.open(img_path).convert('L')
            transformed_img = T.ToTensor()(img)
        elif target_domain == 'svhn': 
            img = Image.open(img_path)
            transformed_img = self.transform(img)
        img.close()
        return transformed_img, self.file_list[idx]

"""## Data Loader"""

import math
img_transform = T.Compose([
    # T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225])
])
hw2_3_test = hw2_3_dataset_val(args.img_dir, img_transform)
BATCH_SIZE = 128
test_loader = DataLoader(hw2_3_test, batch_size=BATCH_SIZE, shuffle=False)

class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN(nn.Module):
    def __init__(self,num_classes=10):
        super(DANN,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(channel,32,5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,48,5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.avgpool=nn.AdaptiveAvgPool2d((5,5))
        self.task_classifier=nn.Sequential(
            nn.Linear(48*5*5,100),
            nn.ReLU(inplace=True),
            nn.Linear(100,100),
            nn.ReLU(inplace=True),
            nn.Linear(100,num_classes)
        )
        self.domain_classifier=nn.Sequential(
            nn.Linear(48*5*5,100),
            nn.ReLU(inplace=True),
            nn.Linear(100,2)
        )
        self.GRL=GRL()
    def forward(self,x,alpha):
        x = x.expand(x.data.shape[0], channel, 28, 28)
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        task_predict=self.task_classifier(x)
        x=GRL.apply(x,alpha)
        domain_predict=self.domain_classifier(x)
        return task_predict,domain_predict

pretrained_model = DANN()
pretrained_model.load_state_dict(torch.load(model_dict, map_location=torch.device('cpu')))
pretrained_model.eval()


with open(args.dst_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_name', 'label'])
    for batch_idx, (imgs, img_name) in enumerate(test_loader):
        outputs, _ = pretrained_model(imgs, alpha = 0.)
        preds = torch.max(outputs, dim=1)[1]
        for i in range(imgs.size()[0]):
            writer.writerow([img_name[i], preds[i].item()])
            # print(img_path[i], preds[i].item(), con)
            # con+=1