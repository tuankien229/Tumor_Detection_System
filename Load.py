# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 22:45:27 2022

@author: tuank
"""

import os
import pandas as pd
import numpy as np
import torchio as tio
import matplotlib.pyplot as plt
import torch
import subprocess
from utils.ImageProcess import ImageProcess

class LoadImage():
    def __init__(self, folder_path, folder_type):
        self.folder_path = folder_path 
        self.folder_type = folder_type
        self.df = {}
    def ReadFolder(self):
        list_folder = []
        for f in os.listdir(self.folder_path):
            if os.path.isfile(os.path.join(self.folder_path, f)) and f.lower().endswith(("t1.nii.gz",".dcm", "t1.nii", ".DCM")):
                list_folder.append(os.path.join(self.folder_path, f))
        return list_folder
    def ReadImage(self):
        list_folder = self.ReadFolder()
        im_process = ImageProcess()
        if self.folder_type == '.nii' or self.folder_type == '.nii.gz':
            subjects = tio.ScalarImage(list_folder[0])
        elif self.folder_type == '.dcm' or self.folder_type == '.DCM':
            w,h,c = self.CheckShapeImage()
            list_img = []
            for f in list_folder:
                subject = tio.ScalarImage(f)
                subject = np.array(subject)
                subject = np.squeeze(subject, 0)
                h_,w_,c_ = subject.shape
                if h_ == h and w_ == w and c_ == c:
                    subject = np.array(subject, dtype=np.uint8)
                    # pre_subject = im_process.RemoveNoise(subject)
                    # pre_subject = im_process.RemoveRice(pre_subject)
                    # pre_subject = im_process.RemoveNoise(pre_subject)
                    # pre_subject = im_process.Threshold(subject)
                    # print(pre_subject.shape)
                    # pre_subject = np.moveaxis(pre_subject[np.newaxis], (0,1,2), (2,0,1))
                    list_img.append(subject)
            subjects = np.stack(list_img)
            subjects = np.moveaxis(subjects, (0,1,2,3), (3,1,2,0))
            subjects = torch.Tensor(subjects)
            subjects = tio.ScalarImage(tensor = subjects)
            os.system('mkdir -p nii_file')
            subjects.save('nii_file/brain_dcm.nii.gz')
            # remove_skull = im_process.RemoveSkull(os.getcwd())
            # subjects = tio.ScalarImage('nii_file/brain_dcm_bet.nii.gz')
        return subjects
    def ShowImage(self):
        subjects = self.ReadImage()
        subjects = subjects[tio.DATA].numpy().squeeze()
        # print(subjects.shape)
        subjects = np.moveaxis(subjects, (0,1,2), (2,1,0))
        x,y,z = subjects.shape
        subjects_1 = np.rot90(subjects[:,:,int(z/2)], 2)
        subjects_2 = np.rot90(subjects[:,int(y/2),:], 2)
        subjects_3 = subjects[int(x/2),:,:]
        list_show = [subjects_1, subjects_2, subjects_3]
        return list_show
    def CheckShapeImage(self):
        list_folder = self.ReadFolder()
        high = []
        weight = []
        channel = []
        for f in list_folder:
            subject = tio.ScalarImage(f)
            subject = np.array(subject)
            subject = np.squeeze(subject, 0)
            high.append(subject.shape[0])
            weight.append(subject.shape[1])
            channel.append(subject.shape[2])
        self.df['Weight'] = weight
        self.df['High'] = high
        self.df['Channel'] = channel
        self.df = pd.DataFrame.from_dict(self.df)
        self.df['Weight'] = self.df.Weight.astype('category')
        self.df['High'] = self.df.High.astype('category')
        self.df['Channel'] = self.df.Channel.astype('category')
        weight = self.df.Weight.value_counts().index.tolist()[0]
        high = self.df.High.value_counts().index.tolist()[0]
        channel = self.df.Channel.value_counts().index.tolist()[0]
        self.df = self.df.loc[self.df['Weight'] == weight].reset_index(drop = True)
        self.df = self.df.loc[self.df['High'] == high].reset_index(drop = True)
        self.df = self.df.loc[self.df['Channel'] == channel].reset_index(drop = True)
        return weight, high, channel
if __name__ == "__main__":
    path_1 = 'D:/Spyder/Tumor_model/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001'
    path_2 = 'D:/Spyder/Image/VU QUANG TAO M15'
    path_3 = 'D:/Spyder/Image/ScalarVolume_14/ScalarVolume_14'
    load_image = LoadImage(folder_path=path_2,folder_type='.dcm')
    list_show = load_image.ShowImage()
    for img in list_show:
        plt.imshow(img)
        plt.show()