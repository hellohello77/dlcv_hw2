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
parser.add_argument("dataset")
args = parser.parse_args()
corrects = 0
total = 0
with open(f'./hw2_data/digits/{args.dataset}/val.csv', newline='') as csvfile2:
    test_bench = []
    reader2 = csv.reader(csvfile2, delimiter=',')
    for idx2, row2 in enumerate(reader2):
        test_bench.append(row2)
with open('test.csv', newline='') as csvfile1:
    reader1 = csv.reader(csvfile1, delimiter=',')
    for idx1, row1 in enumerate(reader1):
        if idx1:
            name = row1[0]
            for row2 in test_bench:
                if name == row2[0]:
                    if row1[1] == row2[1]:
                        corrects += 1
                    total += 1
print(corrects/total)