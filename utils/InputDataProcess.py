# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 09:16:08 2022

@author: tuank
"""
import os
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
class GlobalConfig():
    root = 'D:/Spyder/Tumor_model/'
    root_system = str(os.getcwd())
    train_path = 'MICCAI_BraTS2020_TrainingData/'
    val_path = 'MICCAI_BraTS2020_ValidationData/'
    name_mapping_train = 'name_mapping.csv'
    survival_info_train = 'survival_info.csv'
    name_mapping_test = 'name_mapping_validation_data.csv'
    survival_info_test = 'survival_evaluation.csv'
    check_point = 'best_checkpoint/'
    train_df = 'train_df.csv'
    val_df = 'test_df.csv'
    seed = 55

class ProcessData():
    def SeedEveryThing(self, seed:int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def PreprocessData(self, phase_: str='train'):
        """
        In this dataset of MICCAI_BraTS2020, it has two data files CSV so we need to
        merge them into one data frame to visualize and remove null data 
        """
        key = 'Brats20ID'
        config = GlobalConfig()
        if phase_ == 'train':
            path_ = config.train_path
            save_data_path = config.train_df
            name_mapping = config.name_mapping_train
            survival_info = config.survival_info_train
            name_feature = 'Brats20ID'
        else:
            path_ = config.val_path
            save_data_path = config.val_df
            name_mapping = config.name_mapping_test
            survival_info = config.survival_info_test
            name_feature = 'BraTS20ID'
        
        name_mapping = pd.read_csv(os.path.join(config.root, path_ + name_mapping))
        survival_info = pd.read_csv(os.path.join(config.root, path_ + survival_info))
        name_mapping.rename({'BraTS_2020_subject_ID': key}, axis = 1, inplace = True)
        survival_info.rename({name_feature: key}, axis = 1, inplace = True)
        df = survival_info.merge(name_mapping, on=key, how='right')
        path = []
        for _, row in df.iterrows():
            id_ = row[key]
            phase = id_.split('_')[-2]
            if phase == 'Training':
                data_path = os.path.join(config.root, path_ + id_)
            else:
                data_path = os.path.join(config.root, path_ + id_)
            path.append(data_path)
        df['Path'] = path
        df['Age_rank'] = df['Age'].values//10*10
        df= df.loc[df[key] != 'BraTS20_Training_355'].reset_index(drop = True)
        train_df = df.loc[df['Age'].isnull() != True].reset_index(drop = True)
        skf = StratifiedKFold(n_splits=7, random_state=config.seed, shuffle = True)
        for i, (train_index, val_index) in enumerate(skf.split(train_df, train_df['Age_rank'])):
            train_df.loc[val_index,['Fold']] = i
        train_df.to_csv(os.path.join(config.root_system, config.check_point +  save_data_path), index = False)
    

# Test
if __name__ == '__main__':
    config = GlobalConfig()
    data = ProcessData()
    test = data.PreprocessData('test')