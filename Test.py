# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 17:02:28 2022

@author: tuank
"""


from utils.Load import LoadData, LoadImg
import os
import pandas as pd
import torchio as tio
import numpy as np
if __name__ == '__main__':
    os_path = os.getcwd()
    data_path = 'D:/Spyder/Image/VU QUANG TAO M15'    
    load_data = LoadData(data_path=data_path)
    df = load_data.ReadFolder()
        