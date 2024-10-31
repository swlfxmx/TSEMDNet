import torch
import torch.nn as nn
from torchvision import models
import cv2
import glob
from torch.nn import Softmax
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import os
import time
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder1 = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU())

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())

        self.encoder4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.encoder1(x)  # 112*112*8
        x2 = self.encoder2(x1)  # 56*56*16
        x3 = self.encoder3(x2)  # 28*28*32
        x4 = self.encoder4(x3)  # 14*14*64

        return x4


class SD_900_Class_split(nn.Module):
    def __init__(self):
        super(SD_900_Class_split, self).__init__()
        self.en = CAE()
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        Sc_split = []
        Pa_split = []
        In_split = []
        xe = self.en(x)  # bc*64*14*14
        mean_vector = torch.mean(xe, dim=(2, 3))  # bc*64
        mean_vector1 = self.fc1(mean_vector)  # bc*16
        predict = self.fc2(mean_vector1)  # 预测向量

        softmax_predict = torch.softmax(predict, dim=1)  # softmax处理
        predict_cla = torch.argmax(softmax_predict, dim=1)

        Sc_class_indices = (predict_cla == 0).nonzero().squeeze()
        if Sc_class_indices.numel() > 0:
            if Sc_class_indices.numel() == 1:
                Sc_split = x[Sc_class_indices].unsqueeze(0)
            else:
                Sc_split = x[Sc_class_indices]

        else:
            Sc_split = torch.tensor([], dtype=x.dtype).to('cuda')

        Pa_class_indices = (predict_cla == 1).nonzero().squeeze()
        if Pa_class_indices.numel() > 0:
            if Pa_class_indices.numel() == 1:
                Pa_split = x[Pa_class_indices].unsqueeze(0)
            else:
                Pa_split = x[Pa_class_indices]

        else:
            Pa_split = torch.tensor([], dtype=x.dtype).to('cuda')

        In_class_indices = (predict_cla == 2).nonzero().squeeze()
        if In_class_indices.numel() > 0:
            if In_class_indices.numel() == 1:
                In_split = x[In_class_indices].unsqueeze(0)
            else:
                In_split = x[In_class_indices]
        else:
            In_split = torch.tensor([], dtype=x.dtype).to('cuda')

        return predict, Sc_split, Pa_split, In_split


# class X_SDD_Class_split(nn.Module):
#     def __init__(self):
#         super(X_SDD_Class_split, self).__init__()
#         self.en = CAE()
#         self.fc1 = nn.Linear(64, 16)
#         self.fc2 = nn.Linear(16, 6)
#     def forward(self, x):

#         Frp_split=[]
#         Isa_split=[]
#         Osops_split=[]
#         Ri_split=[]
#         Osots_split=[]
#         Sc_split=[]
#         xe=self.en(x)#bc*64*14*14
#         mean_vector = torch.mean(xe, dim=(2, 3))#bc*64
#         mean_vector1=self.fc1(mean_vector)#bc*16
#         predict = self.fc2(mean_vector1)  # 预测向量

#         softmax_predict = torch.softmax(predict, dim=1)  # softmax处理
#         predict_cla = torch.argmax(softmax_predict, dim=1)

#         Frp_class_indices = (predict_cla == 0).nonzero().squeeze()
#         if Frp_class_indices.numel() > 0:
#             if Frp_class_indices.numel()==1:
#                 Frp_split = x[Frp_class_indices].unsqueeze(0)
#             else:
#                 Frp_split = x[Frp_class_indices]

#         else:
#             binary_Frp_split =torch.tensor([], dtype=x.dtype).to('cuda')

#         Isa_class_indices = (predict_cla == 1).nonzero().squeeze()
#         if Isa_class_indices.numel() > 0:
#             if Isa_class_indices.numel()==1:
#                 Isa_split = x[Isa_class_indices].unsqueeze(0)
#             else:
#                 Isa_split = x[Isa_class_indices]

#         else:
#             binary_Isa_split =torch.tensor([], dtype=x.dtype).to('cuda')

#         Osops_class_indices = (predict_cla == 2).nonzero().squeeze()
#         if Osops_class_indices.numel() > 0:
#             if Osops_class_indices.numel()==1:
#                 Osops_split = x[Osops_class_indices].unsqueeze(0)
#             else:
#                 Osops_split = x[Osops_class_indices]
#         else:
#             binary_Osops_split =torch.tensor([], dtype=x.dtype).to('cuda')


#         Ri_class_indices = (predict_cla == 3).nonzero().squeeze()
#         if Ri_class_indices.numel() > 0:
#             if Ri_class_indices.numel()==1:
#                 Ri_split = x[Ri_class_indices].unsqueeze(0)
#             else:
#                 Ri_split = x[Ri_class_indices]
#         else:
#             binary_Ri_split =torch.tensor([], dtype=x.dtype).to('cuda')


#         Osots_class_indices = (predict_cla == 4).nonzero().squeeze()
#         if Osots_class_indices.numel() > 0:
#             if Osots_class_indices.numel()==1:
#                 Osots_split = x[Osots_class_indices].unsqueeze(0)
#             else:
#                 Osots_split = x[Osots_class_indices]
#         else:
#             binary_Osots_split =torch.tensor([], dtype=x.dtype).to('cuda')


#         Sc_class_indices = (predict_cla == 5).nonzero().squeeze()
#         if Sc_class_indices.numel() > 0:
#             if Sc_class_indices.numel()==1:
#                 Sc_split = x[Sc_class_indices].unsqueeze(0)
#             else:
#                 Sc_split = x[Sc_class_indices]
#         else:
#             binary_Sc_split =torch.tensor([], dtype=x.dtype).to('cuda')

#         return predict,predict_cla,Frp_split, Isa_split, Osops_split,Ri_split,Osots_split,Sc_split


class labelshuffle(nn.Module):
    def __init__(self):
        super(labelshuffle, self).__init__()

    def forward(self, x, y):
        indice = y.tolist()
        #   x_tensor = torch.Tensor(x)

        class_indices = {}
        for i in range(3):
            class_indices[i] = torch.nonzero(torch.tensor(indice) == i).flatten()

        splits = [x[class_indices[i]] for i in range(6)]
        output = torch.cat(splits, dim=0)

        return output

