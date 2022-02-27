# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:54:37 2022

@author: tuank
"""
import os
import torch
import torch.nn as nn
from model.UNET3D import Unet3D, Unet3DPP
from utils.DatasetProcess import BratsDataSet
from utils.LossFunctions import BCEDiceLoss, FocalTverskyLoss
from utils.InputDataProcess import GlobalConfig, ProcessData
from utils.TrainFunction import Training
config = GlobalConfig()
data = ProcessData()
train_data = data.PreprocessData('train')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_1 = Unet3D(in_channels=4, out_channels=32, n_classes=3).to(device)
model = model_1
dataset=BratsDataSet
criterion=BCEDiceLoss()
lr=2.5e-4
num_epochs=200
batch_size = 5
path_to_csv=os.path.join(config.root_system, config.check_point + config.train_df)
fold=0
accumulation_steps=4
save_model_history=True
display_plot=True

trainer=Training(model=model,
                 dataset=dataset,
                 criterion=criterion,
                 lr=lr,
                 num_epochs=num_epochs,
                 batch_size=batch_size,
                 path_to_csv=path_to_csv,
                 fold=fold,
                 accumulation_steps=accumulation_steps,
                 save_model_history=save_model_history,
                 display_plot=display_plot
                )
trainer.run()