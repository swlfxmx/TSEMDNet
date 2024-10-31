import os
import time
import cv2
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import glob
import datetime
from skimage import io
from PIL import Image
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from ptflops import get_model_complexity_info
from torchvision import transforms
from data_loader import RescaleT, ToTensorLab, SalObjDataset
import dataload
# from .resnet_model import *
from torchvision import models
import warnings
import time
import untils

warnings.filterwarnings("ignore",
                        message="The default behavior for interpolate/upsample with float scale_factor changed in 1.6.0 to align with other frameworks/libraries, and now uses scale_factor directly, instead of relying on the computed output size. If you wish to restore the old behavior, please set recompute_scale_factor=True. See the documentation of nn.Upsample for details.")
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
CUDA_LAUNCH_BLOCKING = 1

bce_loss = nn.BCELoss(size_average=True)


def get_multi_loss(pred, target):
    bce_out = bce_loss(pred, target)
    return bce_out


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, depthwise_dilation=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   dilation=depthwise_dilation)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.relu(self.bn(self.pointwise(x)))
        return x


class GLUM(nn.Module):
    def __init__(self, inchannel):
        super(GLUM, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(inchannel, inchannel // 4, kernel_size=(4, 4), stride=2, padding=1,
                                          dilation=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(inchannel, inchannel // 4, kernel_size=(4, 4), stride=2, padding=3,
                                          dilation=2, output_padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(inchannel, inchannel // 4, kernel_size=(4, 4), stride=2, padding=4,
                                          dilation=3, bias=False)

        self.conv1 = DepthwiseSeparableConv2d(inchannel // 4, inchannel // 4, 3, stride=1, padding=1,
                                              depthwise_dilation=1)
        self.conv2 = DepthwiseSeparableConv2d(inchannel // 4, inchannel // 4, 3, stride=1, padding=1,
                                              depthwise_dilation=1)
        self.conv3 = DepthwiseSeparableConv2d(inchannel // 4, inchannel // 4, 3, stride=1, padding=1,
                                              depthwise_dilation=1)
        self.conv4 = DepthwiseSeparableConv2d(inchannel // 4, inchannel // 4, 3, stride=1, padding=1,
                                              depthwise_dilation=1)

    def forward(self, x):
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.deconv3(x)
        xmid1 = x1 + x2
        xmid2 = x2 + x3
        xmid3 = x1 + x3
        xmid4 = self.conv1(xmid1) + (xmid1)
        xmid5 = self.conv2(xmid2) + (xmid2)
        xmid6 = self.conv3(xmid3) + (xmid3)
        xmid7 = xmid4 + xmid5 + xmid6
        y = self.conv4(xmid7) + (xmid7)
        return y


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=4, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    # self.conv1=DepthwiseSeparableConv2d(channel,channel//2,3,1,1)
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class CA(nn.Module):
    def __init__(self, channel, reduction=4, spatial_kernel=7):
        super(CA, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(1, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)

        return channel_out


class SA(nn.Module):
    def __init__(self, channel, reduction=4, spatial_kernel=7):
        super(SA, self).__init__()
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))

        return spatial_out


class PIFM(nn.Module):
    def __init__(self, inchannel):
        super(PIFM, self).__init__()
        self.sa = SA(inchannel // 4)
        self.ca = CA(inchannel // 4)
        self.conv1 = nn.Conv2d(inchannel * 4, inchannel, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fr, fp):
        fr_sattention = self.sa(fr)
        fr_cattention = self.ca(fr)

        fp_sattention = self.sa(fp)
        fp_cattention = self.ca(fp)

        Fd1 = fr * fr_cattention * fr_sattention
        Ff1 = fr * fp_cattention * fp_sattention

        Ff2 = fp * fr_cattention * fr_sattention
        Fp1 = fp * fp_sattention * fp_cattention

        mid = torch.cat((Fd1, Ff1, Ff2, Fp1), dim=0)
        output = self.sigmoid(mid)

        return output


class sc_SDRM(nn.Module):
    def __init__(self):
        super(sc_SDRM, self).__init__()

        self.glum1 = GLUM(256)
        self.glum2 = GLUM(64)
        self.glum3 = GLUM(16)
        self.glum4 = GLUM(4)
        self.PIFM4 = PIFM(4)

        self.encoder1 = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU())

        self.encoder2 = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.encoder3 = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.encoder4 = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU())

        self.decoder = modules.CAE1().decoder
        state_dict_sc = torch.load(r'/root/0130/gt/savepth/twostage/scgtdecoder_198', map_location='cpu')
        self.decoder.load_state_dict(state_dict_sc)
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # 1*224*224#
        x_down1 = self.endocder1(x)
        x_down2 = self.endocder1(x_down1)
        x_down3 = self.endocder1(x_down2)
        x_down4 = self.endocder1(x_down3)

        x_up1 = self.glum1(x_down4)
        x_up2 = self.glum2(x_up1 + x_down3)
        x_up3 = self.glum3(x_up2 + x_down2)
        x_r = self.glum4(x_up3 + x_down1)
        x_p = self.decoder(x_down4)
        output = self.PIFM(x_r, x_p)

        return output


class pa_SDRM(nn.Module):
    def __init__(self):
        super(pa_SDRM, self).__init__()
        self.glum1 = GLUM(256)
        self.glum2 = GLUM(64)
        self.glum3 = GLUM(16)
        self.glum4 = GLUM(4)
        self.PIFM4 = PIFM(4)

        self.encoder1 = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU())

        self.encoder2 = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.encoder3 = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.encoder4 = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU())

        self.decoder = modules.CAE1().decoder
        state_dict_sc = torch.load(r'/root/0130/gt/savepth/twostage/pagtdecoder_99', map_location='cpu')
        self.decoder.load_state_dict(state_dict_sc)
        for p in self.decoder.parameters():
            p.requires_grad = False

    def forward(self, x):  # 1*224*224#
        x_down1 = self.endocder1(x)
        x_down2 = self.endocder1(x_down1)
        x_down3 = self.endocder1(x_down2)
        x_down4 = self.endocder1(x_down3)

        x_up1 = self.glum1(x_down4)
        x_up2 = self.glum2(x_up1 + x_down3)
        x_up3 = self.glum3(x_up2 + x_down2)
        x_r = self.glum4(x_up3 + x_down1)
        x_p = self.decoder(x_down4)
        output = self.PIFM(x_r, x_p)

        return output


class in_SDRM(nn.Module):
    def __init__(self):
        super(in_SDRM, self).__init__()
        self.glum1 = GLUM(256)
        self.glum2 = GLUM(64)
        self.glum3 = GLUM(16)
        self.glum4 = GLUM(4)
        self.PIFM4 = PIFM(4)

        self.encoder1 = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU())

        self.encoder2 = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU())

        self.encoder3 = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.encoder4 = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(5, 5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU())
        self.decoder = modules.CAE1().decoder
        state_dict_sc = torch.load(r'/root/0130/gt/savepth/twostage/lngtdecoder_99', map_location='cpu')
        self.decoder.load_state_dict(state_dict_sc)
        for p in self.decoder.parameters():
            p.requires_grad = False

    def forward(self, x):  # 1*224*224#
        x_down1 = self.endocder1(x)
        x_down2 = self.endocder1(x_down1)
        x_down3 = self.endocder1(x_down2)
        x_down4 = self.endocder1(x_down3)

        x_up1 = self.glum1(x_down4)
        x_up2 = self.glum2(x_up1 + x_down3)
        x_up3 = self.glum3(x_up2 + x_down2)
        x_r = self.glum4(x_up3 + x_down1)
        x_p = self.decoder(x_down4)
        output = self.PIFM(x_r, x_p)
        return output


class TSEMDNet(nn.Module):
    def __init__(self):
        super(TSEMDNet, self).__init__()
        self.dwc1 = DepthwiseSeparableConv2d(3, 64, 3, 1, 1)
        resnet = models.resnet18(pretrained=True)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.cbam1 = modules.CBAMLayer(64)
        self.cbam2 = modules.CBAMLayer(128)
        self.cbam3 = modules.CBAMLayer(256)
        self.cbam4 = modules.CBAMLayer(512)
        self.dwc2 = DepthwiseSeparableConv2d(512, 1024, 3, 2, 1)
        self.conv1 = DepthwiseSeparableConv2d(64, 8, 3, 1, 1)
        self.conv2 = DepthwiseSeparableConv2d(128, 16, 3, 1, 1)
        self.conv3 = DepthwiseSeparableConv2d(256, 32, 3, 1, 1)
        self.conv4 = DepthwiseSeparableConv2d(512, 64, 3, 1, 1)
        self.conv5 = DepthwiseSeparableConv2d(1024, 128, 3, 1, 1)
        self.glum5 = GLUM(128)  # 64*28*28
        self.glum4 = GLUM(64)  # 32
        self.glum3 = GLUM(32)  # 16
        self.glum2 = GLUM(16)  # 8*224*224
        self.conv6 = DepthwiseSeparableConv2d(8, 1, 1)
        self.class_split = untils.SD_900_Class_split()
        self.sc_SDRM = sc_SDRM()
        self.in_SDRM = in_SDRM()
        self.pa_SDRM = pa_SDRM()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        hx = self.dwc1(x)
        h1 = self.encoder1(hx)  # 64, 224*224 ResNet中的前两个3*3卷积
        h1 = self.cbam1(h1)
        h2 = self.encoder2(h1)  # 128, 112*112* ResNet中的第三四个3*3卷积，第三个步长为2，下采样二倍
        h2 = self.cbam2(h2)
        h3 = self.encoder3(h2)  # 256 56*56
        h3 = self.cbam3(h3)
        h4 = self.encoder4(h3)  # 512,28*28
        h4 = self.cbam4(h4)
        h5 = self.dwc2(h4)  # 1024*14*14

        h1_1 = self.conv1(h1)  # 8*224*224
        h2_1 = self.conv2(h2)  # 16*112*12
        h3_1 = self.conv3(h3)  # 32*56*56
        h4_1 = self.conv4(h4)  # 64*28*28
        h5_1 = self.conv5(h5)  # 128*14*14

        f_up1 = self.glum5(h5_1)  # 64*28*28
        f_up2 = self.glum4(f_up1 + h4_1)  # 32*56*56
        f_up3 = self.glum3(f_up2 + h3_1)  # 16*112*112
        f_up4 = self.glum2(f_up3 + h2_1)  # 8*224*224
        initial_featuremap = self.conv6(f_up4)

        ###second stage
        predict, Sc_split, Pa_split, In_split = self.class_split(initial_featuremap)
        if Sc_split.numel() != 0:
            Sc_split = self.sc_SDRM(Sc_split)
        if In_split.numel() != 0:
            In_split = self.in_SDRM(In_split)
        if Pa_split.numel() != 0:
            Pa_split = self.pa_SDRM(Pa_split)

        refine_out = torch.cat((Sc_split, In_split, Pa_split), dim=0)
        return self.sigmoid(initial_featuremap), refine_out, predict


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TSEMDNet()

    model_dir = '/root/0130/gt/encoderpth/'
    epochs = 100
    batch_size = 32
    train_loader = dataload.dataload(batch_size)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0)
    for epoch in range(epochs):
        start_time = time.time()
        epochs = 100
        u = 0
        total_seg_loss = 0
        total_initial_loss = 0
        total_category_loss = 0
        total_loss = 0
        for i, data in enumerate(train_loader, 0):
            inputs, groundtruth, supervision, label = data['image'], data['label'], data['supervision'], data[
                'category']

            inputs = inputs.type(torch.FloatTensor).to(device)
            groundtruth = groundtruth.type(torch.FloatTensor).to(device)
            supervision = supervision.type(torch.FloatTensor).to(device)
            label = label.to(device)

            labelshuffle = untils.labelshuffle()
            gt_refine = labelshuffle(groundtruth, label)

            initial_featuremap, refine_out, predict = model(inputs)
            class_loss = nn.CrossEntropyLoss().to(device)

            category_loss = class_loss(predict, label)  # 分类损失
            initial_loss = get_multi_loss(initial_featuremap, groundtruth)
            seg_loss = get_multi_loss(refine_out, gt_refine)
            loss = initial_loss + category_loss + seg_loss
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_seg_loss += seg_loss.item()
            total_initial_loss += initial_loss.item()
            total_category_loss += category_loss.item()
            total_loss += loss.item()

            # ===================log========================

        print(
            'epoch [{}/{}], total_loss:{:.4f},total_seg_loss:{:.4f},total_category_loss:{:.4f},total_initial_loss:{:.4f}'.format(
                epoch + 1, epochs, total_loss, total_seg_loss, total_category_loss))
        end_time = time.time()  # 记录每一轮训练结束的时间戳
        training_time = end_time - start_time  # 计算每一轮训练的时长
        print(f"Epoch {epoch + 1} training time: {training_time:.2f} seconds")
        print()
        if u > epochs - 10:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch.save(model.state_dict(), model_dir + "twostage_%d" % (epoch))


