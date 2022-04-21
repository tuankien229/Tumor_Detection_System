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
if __name__ == '__main__':
    config = GlobalConfig()
    data = ProcessData()
    train_data = data.PreprocessData('train')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_1 = Unet3D(in_channels=1, out_channels=64, n_classes=1).to(device)
    model_2 = Unet3DPP(in_channels=1, out_channels=48, n_classes=1).to(device)
    model = model_2
    dataset=BratsDataSet
    criterion=FocalTverskyLoss()
    lr=2.5e-4
    num_epochs=200
    batch_size = 5
    path_to_csv=os.path.join(config.root_system, config.check_point + config.train_df)
    fold=0
    accumulation_steps=4
    save_model_history=True
    display_plot=True
    model.load_state_dict(torch.load('D:/Spyder/Tumor_System/best_checkpoint/highest_checkpoint_0.761.pth'))
    model.eval()
    # trainer=Training(model=model,
    #                  dataset=dataset,
    #                  criterion=criterion,
    #                  lr=lr,
    #                  num_epochs=num_epochs,
    #                  batch_size=batch_size,
    #                  path_to_csv=path_to_csv,
    #                  fold=fold,
    #                  accumulation_steps=accumulation_steps,
    #                  save_model_history=save_model_history,
    #                  display_plot=display_plot
    #                 )
    # trainer.run()
