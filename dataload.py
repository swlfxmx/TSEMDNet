import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from skimage import io
from ptflops import get_model_complexity_info
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import sys
import cv2
import numpy as np
import json
import pytorch_ssim
import pytorch_iou
import glob
from data_loader import RescaleT, RandomCrop, ToTensorLab, SalObjDataset


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def dataload(batch):
    tra_image_dir = '/root/data/train/new train/'
    tra_label_dir = '/root/data/train/new train gt/'
    tra_supervision_dir = '/root/data/train/new train gt/'

    #     tra_image_dir = '/root/data/xsdd/0214/xsdd train/'
    #     tra_label_dir =  '/root/data/xsdd/0214/xsdd train gt/'
    #     tra_supervision_dir =  '/root/data/xsdd/0214/xsdd train gt/'

    #     tra_image_dir = '/root/data/pavement crack datasets/new train tr/'
    #     tra_label_dir =  '/root/data/pavement crack datasets/new train gt tr/'
    #     tra_supervision_dir = '/root/data/pavement crack datasets/new train gt tr/'

    image_ext = '.bmp'
    #     image_ext = '.png'
    label_ext = '.png'
    supervision_ext = '.png'

    tra_img_name_list = glob.glob(tra_image_dir + '*' + image_ext)
    tra_supervision_name_list = glob.glob(tra_supervision_dir + '*' + supervision_ext)

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split("/")[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]
        imidx = imidx.split("\\")[-1]
        tra_lbl_name_list.append(tra_label_dir + imidx + label_ext)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("supervision labels: ", len(tra_supervision_name_list))
    print("---")
    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        supervision_name_list=tra_supervision_name_list,
        batch_size=batch,
        transform=transforms.Compose([RescaleT(224), ToTensorLab(flag=0)])

    )

    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch, shuffle=True, num_workers=0)
    # train_bar = tqdm(salobj_dataloader, file=sys.stdout)
    # for step, data in enumerate(train_bar):
    # inputs, groundtruth, supervision,label = data['image'], data['label'], data['supervision'], data['category']
    # print(label)

    return salobj_dataloader


def testdataload(batch):
    tra_image_dir = '/root/data/test/new test/'
    tra_label_dir = '/root/data/test/new test gt/'
    tra_supervision_dir = '/root/data/test/new test gt/'

    #     tra_image_dir = '/root/data/xsdd/0214/xsdd train/'
    #     tra_label_dir =  '/root/data/xsdd/0214/xsdd train gt/'
    #     tra_supervision_dir =  '/root/data/xsdd/0214/xsdd train gt/'

    #     tra_image_dir = '/root/data/pavement crack datasets/new train tr/'
    #     tra_label_dir =  '/root/data/pavement crack datasets/new train gt tr/'
    #     tra_supervision_dir = '/root/data/pavement crack datasets/new train gt tr/'

    image_ext = '.bmp'
    # image_ext = '.png'
    label_ext = '.png'
    supervision_ext = '.png'

    tra_img_name_list = glob.glob(tra_image_dir + '*' + image_ext)
    tra_supervision_name_list = glob.glob(tra_supervision_dir + '*' + supervision_ext)

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split("/")[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]
        imidx = imidx.split("\\")[-1]
        tra_lbl_name_list.append(tra_label_dir + imidx + label_ext)

    print("---")
    print("test images: ", len(tra_img_name_list))
    print("test labels: ", len(tra_lbl_name_list))
    print("test labels: ", len(tra_supervision_name_list))
    print("---")
    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        supervision_name_list=tra_supervision_name_list,
        batch_size=batch,
        transform=transforms.Compose([RescaleT(224), ToTensorLab(flag=0)])

    )

    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch, shuffle=True, num_workers=0)
    # train_bar = tqdm(salobj_dataloader, file=sys.stdout)
    # for step, data in enumerate(train_bar):
    # inputs, groundtruth, supervision,label = data['image'], data['label'], data['supervision'], data['category']
    # print(label)

    return salobj_dataloader
