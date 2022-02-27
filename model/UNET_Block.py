# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 23:56:47 2022

@author: tuank
"""

import torch
import torch.nn as nn
from torch.nn.functional import pad, sigmoid, binary_cross_entropy
from torch.utils.data import DataLoader, Dataset
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.double_conv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        x = self.encoder(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Make sure size data when you combine are the same size
        diffX = x2.size()[4] - x1.size()[4]
        diffY = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[2] - x1.size()[2]
        
        x1 = torch.nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class UpP(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels*3, out_channels)
    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        # Make sure size data when you combine are the same size
        diffX1 = x3.size()[4] - x1.size()[4]
        diffY1 = x3.size()[3] - x1.size()[3]
        diffZ1 = x3.size()[2] - x1.size()[2]
        
        diffX2 = x3.size()[4] - x2.size()[4]
        diffY2 = x3.size()[3] - x2.size()[3]
        diffZ2 = x3.size()[2] - x2.size()[2]
        
        x2 = torch.nn.functional.pad(x2, (diffX2 // 2, diffX2 - diffX2 // 2, diffY2 // 2, diffY2 - diffY2 // 2, diffZ2 // 2, diffZ2 - diffZ2 // 2))
        x1 = torch.nn.functional.pad(x1, (diffX1 // 2, diffX1 - diffX1 // 2, diffY1 // 2, diffY1 - diffY1 // 2, diffZ1 // 2, diffZ1 - diffZ1 // 2))
        x = torch.cat([x3, x2, x1], dim=1)
        return self.conv(x)
    
class UpPP(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels*4, out_channels)
    def forward(self, x1, x2, x3, x4):
        x1 = self.up(x1)
        # Make sure size data when you combine are the same size
        diffX1 = x4.size()[4] - x1.size()[4]
        diffY1 = x4.size()[3] - x1.size()[3]
        diffZ1 = x4.size()[2] - x1.size()[2]
        
        diffX2 = x4.size()[4] - x2.size()[4]
        diffY2 = x4.size()[3] - x2.size()[3]
        diffZ2 = x4.size()[2] - x2.size()[2]
        
        diffX3 = x4.size()[4] - x3.size()[4]
        diffY3 = x4.size()[3] - x3.size()[3]
        diffZ3 = x4.size()[2] - x3.size()[2]
        
        x3 = torch.nn.functional.pad(x3, (diffX3 // 2, diffX3 - diffX3 // 2, diffY3 // 2, diffY3 - diffY3 // 2, diffZ3 // 2, diffZ3 - diffZ3 // 2))
        x2 = torch.nn.functional.pad(x2, (diffX2 // 2, diffX2 - diffX2 // 2, diffY2 // 2, diffY2 - diffY2 // 2, diffZ2 // 2, diffZ2 - diffZ2 // 2))
        x1 = torch.nn.functional.pad(x1, (diffX1 // 2, diffX1 - diffX1 // 2, diffY1 // 2, diffY1 - diffY1 // 2, diffZ1 // 2, diffZ1 - diffZ1 // 2))
        x = torch.cat([x4, x3, x2, x1], dim=1)
        return self.conv(x)
    
class UpPPP(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels*5, out_channels)
    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.up(x1)
        # Make sure size data when you combine are the same size
        diffX1 = x5.size()[4] - x1.size()[4]
        diffY1 = x5.size()[3] - x1.size()[3]
        diffZ1 = x5.size()[2] - x1.size()[2]
        
        diffX2 = x5.size()[4] - x2.size()[4]
        diffY2 = x5.size()[3] - x2.size()[3]
        diffZ2 = x5.size()[2] - x2.size()[2]
        
        diffX3 = x5.size()[4] - x3.size()[4]
        diffY3 = x5.size()[3] - x3.size()[3]
        diffZ3 = x5.size()[2] - x3.size()[2]
        
        diffX4 = x5.size()[4] - x4.size()[4]
        diffY4 = x5.size()[3] - x4.size()[3]
        diffZ4 = x5.size()[2] - x4.size()[2]
        
        x4 = torch.nn.functional.pad(x4, (diffX4 // 2, diffX4 - diffX4 // 2, diffY4 // 2, diffY4 - diffY4 // 2, diffZ4 // 2, diffZ4 - diffZ4 // 2))        
        x3 = torch.nn.functional.pad(x3, (diffX3 // 2, diffX3 - diffX3 // 2, diffY3 // 2, diffY3 - diffY3 // 2, diffZ3 // 2, diffZ3 - diffZ3 // 2))
        x2 = torch.nn.functional.pad(x2, (diffX2 // 2, diffX2 - diffX2 // 2, diffY2 // 2, diffY2 - diffY2 // 2, diffZ2 // 2, diffZ2 - diffZ2 // 2))
        x1 = torch.nn.functional.pad(x1, (diffX1 // 2, diffX1 - diffX1 // 2, diffY1 // 2, diffY1 - diffY1 // 2, diffZ1 // 2, diffZ1 - diffZ1 // 2))
        x = torch.cat([x5, x4, x3, x2, x1], dim=1)
        return self.conv(x)
    
class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.out_conv(x)

