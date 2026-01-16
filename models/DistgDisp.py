import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class Net(nn.Module):
    def __init__(self, angRes):
        super(Net, self).__init__()
        feaC = 16
        channel = 160
        mindisp, maxdisp = -4, 4
        self.angRes = angRes
        self.init_feature = nn.Sequential(
            nn.Conv2d(1, feaC, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feaC, feaC, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            nn.Conv2d(feaC, feaC, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feaC, feaC, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
        )

        self.build_costvolume = BuildCostVolume(feaC, channel, angRes, mindisp, maxdisp)

        self.aggregation = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channel),
            nn.LeakyReLU(0.1, inplace=True),
            ResB3D(channel),
            ResB3D(channel),
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channel, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.regression = Regression(mindisp, maxdisp)

    def forward(self, x):
        x = SAI2MacPI(x, self.angRes)
        init_feat = self.init_feature(x)
        cost = self.build_costvolume(init_feat)
        cost = self.aggregation(cost)
        init_disp = self.regression(cost.squeeze(1))

        return init_disp


class BuildCostVolume(nn.Module):
    def __init__(self, channel_in, channel_out, angRes, mindisp, maxdisp):
        super(BuildCostVolume, self).__init__()
        self.DSAFE = nn.Conv2d(channel_in, channel_out, angRes, stride=angRes, padding=0, bias=False)
        self.angRes = angRes
        self.mindisp = mindisp
        self.maxdisp = maxdisp

    def forward(self, x):
        cost_list = []
        for d in range(self.mindisp, self.maxdisp + 1):
            if d < 0:
                dilat = int(abs(d) * self.angRes + 1)
                pad = int(0.5 * self.angRes * (self.angRes - 1) * abs(d))
            if d == 0:
                dilat = 1
                pad = 0
            if d > 0:
                dilat = int(abs(d) * self.angRes - 1)
                pad = int(0.5 * self.angRes * (self.angRes - 1) * abs(d) - self.angRes + 1)
            cost = F.conv2d(x, weight=self.DSAFE.weight, stride=self.angRes, dilation=dilat, padding=pad)
            cost_list.append(cost)
        cost_volume = torch.stack(cost_list, dim=2)

        return cost_volume


class Regression(nn.Module):
    def __init__(self, mindisp, maxdisp):
        super(Regression, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.maxdisp = maxdisp
        self.mindisp = mindisp

    def forward(self, cost):
        score = self.softmax(cost)  # B, D, H, W
        temp = torch.zeros(score.shape).to(score.device)  # B, D, H, W
        for d in range(self.maxdisp - self.mindisp + 1):
            temp[:, d, :, :] = score[:, d, :, :] * (self.mindisp + d)
        disp = torch.sum(temp, dim=1, keepdim=True)  # B, 1, H, W

        return disp


class SpaResB(nn.Module):
    def __init__(self, channels, angRes):
        super(SpaResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        buffer = self.body(x)
        return buffer + x


class ResB3D(nn.Module):
    def __init__(self, channels):
        super(ResB3D, self).__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )

    def forward(self, x):
        buffer = self.body(x)
        return buffer + x


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    
    # 原始逻辑是重排像素，可以用 reshape + permute 实现，速度更快且支持 ONNX 动态尺寸
    # 假设输入是 [B, C, angRes*h, angRes*w]
    # 1. 拆分维度: [B, C, angRes, h, angRes, w]
    x = x.view(b, c, angRes, h, angRes, w)
    
    # 2. 交换维度，把 h, w 移到一起，angRes 移到一起
    # 目标顺序: [B, C, h, angRes, w, angRes]
    x = x.permute(0, 1, 3, 2, 5, 4)
    
    # 3. 合并维度: [B, C, h*angRes, w*angRes]
    # 注意：虽然输出 shape 和输入一样，但内存里的排列顺序变了（完成了 MacPI 重组）
    out = x.reshape(b, c, hu, wv)
    
    return out


import utils

if __name__ == "__main__":
    ANG_RES = 9  
    MODEL_PATH = "models/DistgDisp.pth.tar" 
    ONNX_PATH = f"models/DistgDisp_{ANG_RES}x{ANG_RES}.onnx"

    net = Net(angRes=ANG_RES).cpu()
    utils.load_checkpoint(net, MODEL_PATH)
    utils.export_to_onnx(net, ONNX_PATH, ANG_RES)