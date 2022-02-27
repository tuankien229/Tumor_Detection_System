# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 22:09:41 2022

@author: tuank
"""
import os
import pandas as pd
import numpy as np
import torchio as tio
import matplotlib.pyplot as plt
import torch
class LoadData():
    """ Abstract class for load the data
    - ReadFolder(self, data_path): read and find all dicom file or nii file 
    in the folder and save it into the database
    - CheckTypeOfPath(self, file): check type of file 
    - GetInfo(self, file): get info about file like id, type of file 
    - ReadFile(self, file): read file and return path of the file affter 
    transform to right format for loading image    
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.database = {'ID': [], 'Type': [], 'Path': []}
        self.list_type_dicom = ['DCM', 'dcm']
        self.list_type_nii = ['nii', 'gz']
        self.id = None
        self.type = None
    def ReadFolder(self):
        for file in os.listdir(data_path):
            self.GetInfo(file)
            if self.CheckTypeOfFile():
                path = self.data_path + '/' + self.id + self.type
                self.database['ID'].append(self.id)
                self.database['Type'].append(self.type)
                self.database['Path'].append(path)
            else:
                pass
        df = pd.DataFrame.from_dict(self.database)
        return df
    def CheckTypeOfFile(self):
        if self.type != None and self.id != None:
            return True
        return False
    def GetInfo(self, file):
        type_ = file.split('.')[-1]
        for i in range(2):
            if type_ == self.list_type_nii[i]:
                if type_ == 'gz':
                    if file.split('nii')[-1] == '.gz':
                        type_ = 'nii.gz'
                    else:
                        break
                self.type = '.nii.gz'
                if file.split('t1')[-1] == self.type or file.split('T1')[-1] == self.type:
                    self.id = file.split('.'+type_)[0]
                else:
                    self.id = None
            elif type_ == self.list_type_dicom[i]:
                self.id = file.split(str('.'+type_))[0]
                self.type = '.dcm'
            

class LoadImg():
    def __init__(self, df:pd.DataFrame):
        self.df = df
        self.row_size = 0
        self.col_size = 0
    def CheckDatabase(self):
        self.GetInfo()
        for i in range(len(self.df.High.cat.categories)):
            detect_max_high = 0
            max_counts = self.df.High.value_counts()[0]
            if self.df.High.value_counts()[i] > max_counts:
                detect_max_high = i
                max_counts = self.df.High.value_counts()[i]  
        for i in range(len(self.df.Weight.cat.categories)):
            detect_max_weight = 0
            max_counts = self.df.Weight.value_counts()[0]
            if self.df.Weight.value_counts()[i] > max_counts:
                detect_max_weight = i
                max_counts = self.df.High.value_counts()[i]
        if len(self.df.High.cat.categories) < len(self.df.Weight.cat.categories):
            self.df = self.df.loc[self.df['High'] == self.df.High.cat.categories[detect_max_high]].reset_index(drop = True)
            self.df = self.df.loc[self.df['Weight'] == self.df.Weight.cat.categories[detect_max_weight]].reset_index(drop = True)
        else:
            self.df = self.df.loc[self.df['Weight'] == self.df.Weight.cat.categories[detect_max_weight]].reset_index(drop = True)
            self.df = self.df.loc[self.df['High'] == self.df.High.cat.categories[detect_max_high]].reset_index(drop = True)
        return self.df
    def StackImage(self):
        self.CheckDatabase()
        dicom_data = self.df.loc[self.df['Type'] == '.dcm'].reset_index(drop = True)
        nii_data = self.df.loc[self.df['Type'] == '.nii.gz'].reset_index(drop = True)
        if len(dicom_data) > 0:
            list_img = []
            for i, id_ in enumerate(dicom_data['ID']):
                path = dicom_data.loc[dicom_data['ID'] == id_]['Path']
                subject = tio.ScalarImage(path)
                subject = np.array(subject)
                subject = np.squeeze(subject, 0)
                list_img.append(subject)
            subjects = np.stack(list_img)
            subjects = np.moveaxis(subjects, (0,1,2,3), (3,1,2,0))
            subjects = torch.Tensor(subjects)
            subjects = tio.ScalarImage(tensor = subjects)
            return subjects
        for i, id_ in enumerate(nii_data['ID']):
            path = nii_data.loc[nii_data['ID'] == id_]['Path']
            subjects = tio.ScalarImage(path)
        return subjects
        
    def GetInfo(self):
        high = []
        weight = []
        channel = []
        for i, id_ in enumerate(self.df['ID']):
            path = self.df.loc[self.df['ID'] == id_]['Path']
            subject = tio.ScalarImage(path)
            subject = np.array(subject)
            subject = np.squeeze(subject, 0)
            high.append(subject.shape[0])
            weight.append(subject.shape[1])
            channel.append(subject.shape[2])
        self.df['High'] = high
        self.df['High'] = self.df.High.astype('category')
        self.df['Weight'] = weight
        self.df['Weight'] = self.df.Weight.astype('category')
        self.df['Channel'] = channel
        self.df['Channel'] = self.df.Channel.astype('category')
    # def ImgProcess(self, img):
        
if __name__ == '__main__':
    os_path = os.getcwd()
    data_path = 'D:/Spyder/Tumor_model/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001'    
    load_data = LoadData(data_path=data_path)
    df = load_data.ReadFolder()
    load_img = LoadImg(df)
    test = load_img.StackImage()
# if __name__ == '__main__':
#     os_path = os.getcwd()
#     data_path = 'D:/Spyder/Image/VU QUANG TAO M15'
#     database = {'ID':[], 'Type':[], 'Path': []}
#     for file in os.listdir(data_path):
#         load_data = LoadData(file)
#         id_, type_, name_ = load_data.ReadFile()
#         if id_ != None and type_ != None and name_ != None:
#             path_ = data_path + '/' + name_
#             database['ID'].append(id_)
#             database['Type'].append(type_)
#             database['Path'].append(path_)
#     df = pd.DataFrame.from_dict(database)
#     for i, id_ in enumerate(df['ID']):
#         path = df.loc[df['ID'] == id_]['Path']
#         print(path)
            