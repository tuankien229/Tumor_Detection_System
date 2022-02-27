# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 09:14:20 2022

@author: tuank
"""

import os
import sys
sys.path.append(r"D:/Spyder/Tumor_System")
import torch
import torchio as tio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
# from utils.InputDataProcess import GlobalConfig
import warnings
warnings.filterwarnings('ignore')
class BratsDataSet(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str = 'test'):
        self.df = df
        self.phase = phase
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index):
        id_ = self.df.loc[index, 'Brats20ID']
        data_path = self.df.loc[self.df['Brats20ID'] == id_]['Path'].values[0]
        resample = tio.Resample((2,2,2))
        img = tio.ScalarImage(os.path.join(data_path, id_ + '_t1.nii.gz')) # data_img shape (1, 240, 240, 155)
        img = resample(img) # data_img shape (1, 120, 120, 78)
        img = np.array(img)
        img = np.squeeze(img, axis = 0)
        img = self.Normalize(img)
        img_stack = np.moveaxis(img, (0,1,2), (0,2,1))
        img_stack = torch.Tensor(img_stack)
        img_stack = torch.unsqueeze(img_stack, dim = 0)
        
        if self.phase != 'test':
            labels = tio.LabelMap(os.path.join(data_path, id_ + '_seg.nii.gz'))
            labels = resample(labels)
            labels = np.array(labels)
            labels = np.squeeze(labels, axis = 0)
            label_stack = self.ConvertToMultiChannel(labels)
            label_stack = torch.Tensor(label_stack)
            label_stack = torch.unsqueeze(label_stack, dim = 0)
            subjects = tio.Subject(image = tio.ScalarImage(tensor = img_stack),
                                   label = tio.LabelMap(tensor = (label_stack > 0.5)),
                                   id = id_
                                  )
            return subjects
        subjects = tio.Subject(image = tio.ScalarImage(tensor = img_stack),
                               id = id_
                              )
        return subjects
    
    def Normalize(self, image : np.ndarray):
        return (image - np.min(image))/(np.max(image) - np.min(image))
 
    def ConvertToMultiChannel(self, labels):
        '''
        Convert labels to multi channels based on brats classes:
        label 1 is the peritumoral edema
        label 2 is the GD-enhancing tumor
        label 3 is the necrotic and non-enhancing tumor core
        The possible classes are TC (Tumor core), WT (Whole tumor)
        and ET (Enhancing tumor)
        '''
        label_WT = labels.copy()
        label_WT[label_WT == 1] = 1
        label_WT[label_WT == 2] = 1
        label_WT[label_WT == 4] = 1
        label_stack = np.moveaxis(label_WT, (0,1,2), (0,2,1))
        return label_stack

class GetData():
    # List transform
    def get_transform(self, phase):
        if phase == 'train':
            list_transforms = [
                tio.RandomAffine(p = 0.5),
                tio.RandomFlip(axes=['LR', 'AP', 'IS'], p = 0.25),          
            ]
            transform = tio.Compose(list_transforms)
        else:
            list_transforms = []
            transform = tio.Compose(list_transforms)
        return transform
    
    
    # Get DataLoader
    def get_dataloader(self, dataset, path_to_csv, phase, fold = 0, batch_size = 1, num_workers = 4):
        """
        This  function is saving image data in to the list and putting it with torchio.SubjectDataSet
        to split and transform image
        """
        data = pd.read_csv(path_to_csv)
        train_data = data.loc[data['Fold'] != fold].reset_index(drop = True)
        val_data = data.loc[data['Fold'] == fold].reset_index(drop = True)
        if phase == 'train':
            data = train_data
        elif phase == 'valid':
            data = val_data
        data_set = dataset(data, phase)
        list_subjects = []
        for i in range(len(data_set)):
            list_subjects.append(data_set[i])
        subject_dataset = tio.SubjectsDataset(list_subjects, transform=self.get_transform(phase))
        patch_size = 50
        queue_length = 300
        sample_per_volume = 1
        sampler = tio.data.UniformSampler(patch_size)
        patches_queue = tio.Queue(
            subject_dataset,
            queue_length,
            sample_per_volume,
            sampler,
            num_workers=num_workers,
        )
        data_loader = DataLoader(patches_queue,
                                 batch_size = batch_size,
                                 num_workers=0,
                                 pin_memory=True,
                                )
        print('Finish Load DataSet')
        return data_loader
# Test
# if __name__ == '__main__':
#     config = GlobalConfig()
#     getdata = GetData()
#     dataloader = getdata.get_dataloader(dataset=BratsDataSet, path_to_csv=config.train_df, phase='valid', fold=0)
#     len(dataloader)
    