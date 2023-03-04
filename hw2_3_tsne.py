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
from sklearn.manifold import TSNE

class hw2_3_dataset_val:
    def __init__(self, filepath, transform, target_domain):
        self.transform = transform
        self.filepath = filepath
        self.file_list = []
        self.labels = []
        self.target_domain = target_domain
        with open(os.path.join(filepath, 'val.csv'), newline='') as csvfile:
          reader = csv.reader(csvfile, delimiter=',')
          for idx, row in enumerate(reader):
            if idx:
              self.file_list.append(row[0])
              self.labels.append(int(row[1]))
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.filepath, 'data', self.file_list[idx])
        if self.target_domain == 'usps':
            img = Image.open(img_path).convert('L')
            transformed_img = T.ToTensor()(img)
        elif self.target_domain == 'svhn': 
            img = Image.open(img_path)
            transformed_img = self.transform(img)
        img.close()
        return transformed_img, self.labels[idx]

"""## Data Loader"""

import math
img_transform = T.Compose([
    # T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225])
])
hw2_3_svhn = hw2_3_dataset_val('./hw2_data/digits/svhn', img_transform, target_domain = 'svhn')
hw2_3_usps = hw2_3_dataset_val('./hw2_data/digits/usps', img_transform, target_domain = 'usps')
hw2_3_mnist1 = hw2_3_dataset_val('./hw2_data/digits/mnistm', img_transform, target_domain = 'svhn')
hw2_3_mnist2 = hw2_3_dataset_val('./hw2_data/digits/mnistm', img_transform, target_domain = 'usps')
BATCH_SIZE = 128
svhn_loader = DataLoader(hw2_3_svhn, batch_size=BATCH_SIZE, shuffle=False)
usps_loader = DataLoader(hw2_3_usps, batch_size=BATCH_SIZE, shuffle=False)
mnist_loader1 = DataLoader(hw2_3_mnist1, batch_size=BATCH_SIZE, shuffle=False)
mnist_loader2 = DataLoader(hw2_3_mnist2, batch_size=BATCH_SIZE, shuffle=False)

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
    def __init__(self,num_classes=10,channel = 3):
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
        self.channel = channel
    def forward(self,x,alpha):
        x = x.expand(x.data.shape[0], self.channel, 28, 28)
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        task_predict=self.task_classifier(x)
        x=GRL.apply(x,alpha)
        domain_predict=self.domain_classifier(x)
        return task_predict,domain_predict

usps_model = DANN(channel=1)
usps_model.load_state_dict(torch.load('./DANN_test11_usps_81.ckpt', map_location=torch.device('cpu')))
usps_model.eval()

svhn_model = DANN()
svhn_model.load_state_dict(torch.load('./DANN_39_504.ckpt', map_location=torch.device('cpu')))
svhn_model.eval()
class_features = []
# datasets = ['usps', 'svhn']
datasets = ['usps']
with torch.no_grad():
    def save_features(module, fin, fout):
        global class_features
        fs = torch.flatten(fout, 1)
        class_features.append(fs.detach().numpy())
    for idx, dataset in enumerate(datasets):
        # if idx == 0:
        #     continue
        print('start')
        class_features = []
        class_labels = []
        domain_labels = []
        

        if idx:
            print('model')
            model = svhn_model
            print('loader')
            test_loader = svhn_loader
            mnist_loader = mnist_loader1
        else:
            model = usps_model
            test_loader = usps_loader
            mnist_loader = mnist_loader2
        handler = model.task_classifier[3].register_forward_hook(save_features)
        k = 0
        for imgs, labels in test_loader:
            print(k)
            k+=1
            _ = model(imgs, 0)
            class_labels.append(labels.detach().numpy())
            domain_labels.append(np.zeros((imgs.size()[0])))

        for imgs, labels in mnist_loader:
            _ = model(imgs, 0)
            class_labels.append(labels.detach().numpy())
            domain_labels.append(np.ones((imgs.size()[0])))

        handler.remove()

        features = np.concatenate(class_features)
        labels = np.concatenate(class_labels)
        domains = np.concatenate(domain_labels)

        tsne = TSNE(n_components=2, perplexity=50)
        reduced = tsne.fit_transform(features)

        plt.figure(figsize=(16, 8))

        for i in range(10):
            selected = reduced[np.where(labels == i)[0]]
            plt.scatter(selected[:, 0], selected[:, 1], label=str(i))
            
        plt.tight_layout()
        plt.legend()
        plt.show()
        # plt.savefig(f'{dataset}_0.png')

        plt.figure(figsize=(16, 8))

        for i in range(2):
            selected = reduced[np.where(domains == i)[0]]
            plt.scatter(selected[:, 0], selected[:, 1], label=str(i))
            
        plt.tight_layout()
        plt.legend()
        plt.show()
        # plt.savefig(f'{dataset}_1.png')
        print('first')