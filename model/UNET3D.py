# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 23:54:55 2022

@author: tuank
"""
import torch
import torch.nn as nn
from torch.nn.functional import pad, sigmoid, binary_cross_entropy
from torch.utils.data import DataLoader, Dataset
from model.UNET_Block import DoubleConv, Down, Up, UpP, UpPP, UpPPP, Out

class Unet3D(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes
        
        self.double_conv = DoubleConv(self.in_channels, self.out_channels)
        # encoder downsamplers
        self.down_1 = Down(self.out_channels, self.out_channels*2)
        self.down_2 = Down(self.out_channels*2, self.out_channels*4)
        self.down_3 = Down(self.out_channels*4, self.out_channels*8)
        self.down_4 = Down(self.out_channels*8, self.out_channels*16)

        # decoder upsamplers
        self.up_4 = Up(self.out_channels*16, self.out_channels*8)
        self.up_3 = Up(self.out_channels*8, self.out_channels*4)
        self.up_2 = Up(self.out_channels*4, self.out_channels*2)
        self.up_1 = Up(self.out_channels*2, self.out_channels)

        # output
        self.out = Out(self.out_channels, self.n_classes)
        
        self.dropout = nn.Dropout3d(0.5)
    def forward(self, x):
        # Encoder
        en_1 = self.double_conv(x)
        en_2 = self.down_1(en_1)
        en_2 = self.dropout(en_2)
        en_3 = self.down_2(en_2)
        en_4 = self.down_3(en_3)
        en_4 = self.dropout(en_4)
        en_5 = self.down_4(en_4)
        
        # Decoder
        de_4 = self.up_4(en_5, en_4)
        de_4 = self.dropout(de_4)
        de_3 = self.up_3(de_4, en_3)
        de_2 = self.up_2(de_3, en_2)
        de_2 = self.dropout(de_2)
        de_1 = self.up_1(de_2, en_1)
        
        out = self.out(de_1)
        return out

class Unet3DPP(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_classes = n_classes
        
        self.x00 = DoubleConv(self.in_channels, self.out_channels)
        # UNet3D ++ L1
        self.down_to_x10 = Down(self.out_channels, self.out_channels*2)
        self.up_to_x01 = Up(self.out_channels*2, self.out_channels)
        # UNet3D ++ L2
        self.down_to_x20 = Down(self.out_channels*2, self.out_channels*4)
        self.up_to_x11 = Up(self.out_channels*4, self.out_channels*2)
        self.up_to_x02 = UpP(self.out_channels*2, self.out_channels)
        # UNet3D ++ L3
        self.down_to_x30 = Down(self.out_channels*4, self.out_channels*8)
        self.up_to_x21 = Up(self.out_channels*8, self.out_channels*4)
        self.up_to_x12 = UpP(self.out_channels*4, self.out_channels*2)
        self.up_to_x03 = UpPP(self.out_channels*2, self.out_channels)
        # UNet3D ++ L4
        self.down_to_x40 = Down(self.out_channels*8, self.out_channels*16)
        self.up_to_x31 = Up(self.out_channels*16, self.out_channels*8)
        self.up_to_x22 = UpP(self.out_channels*8, self.out_channels*4)
        self.up_to_x13 = UpPP(self.out_channels*4, self.out_channels*2)
        self.up_to_x04 = UpPPP(self.out_channels*2, self.out_channels)
        
        # output
        self.out = Out(self.out_channels, self.n_classes)
        
        self.dropout = nn.Dropout3d(0.25)
    def forward(self, x):
        x00 = self.x00(x)
        # UNet3D ++ L1
        x10 = self.down_to_x10(x00)
        x01 = self.up_to_x01(x10, x00)
        x01 = self.dropout(x01)
        # UNet3D ++ L2
        x20 = self.down_to_x20(x10)
        x11 = self.up_to_x11(x20, x10)
        x11 = self.dropout(x11)
        x02 = self.up_to_x02(x11, x01, x00)
        x02 = self.dropout(x02)
        # UNet3D ++ L3
        x30 = self.down_to_x30(x20)
        x21 = self.up_to_x21(x30, x20)
        x21 = self.dropout(x21)
        x12 = self.up_to_x12(x21, x11, x10)
        x12 = self.dropout(x12)
        x03 = self.up_to_x03(x12, x02, x01, x00)
        x03 = self.dropout(x03)
        # UNet3D ++ L4
        x40 = self.down_to_x40(x30)
        x31 = self.up_to_x31(x40, x30)
        x31 = self.dropout(x31)
        x22 = self.up_to_x22(x31, x21, x20)
        x22 = self.dropout(x22)
        x13 = self.up_to_x13(x22, x12, x11, x10)
        x13 = self.dropout(x13)
        x04 = self.up_to_x04(x13, x03, x02, x01, x00)
        x04 = self.dropout(x04)
        
        # Output
        out = self.out(x04)
        return out
# if __name__ == '__main__':
#     model = UNET3D(in_channels=1, out_channels=32, n_classes=1).to('cuda')
#     print(model.eval())