# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 22:09:41 2022

@author: tuank
"""
import os
import pandas as pd
class LoadData():
    """ Abstract class for load the data
    - ReadFolder(self, data_path): read and find all dicom file or nii file 
    in the folder and save it into the database
    - CheckTypeOfPath(self, file): check type of file 
    - GetInfo(self, file): get info about file like id, type of file 
    - ReadFile(self, file): read file and return path of the file affter 
    transform to right format for loading image    
    """
    def __init__(self, file):
        self.file = file
        self.list_type_dicom = ['DCM', 'dcm']
        self.list_type_nii = ['nii', 'gz']
        self.id = None
        self.type = None
    def CheckTypeOfFile(self):
        if self.type != None:
            return True
        return False
    def GetInfo(self):
        type_ = self.file.split('.')[-1]
        for i in range(2):
            if type_ == self.list_type_nii[i]:
                if type_ == 'gz':
                    if self.file.split('nii')[-1] == '.gz':
                        type_ = 'nii.gz'
                    else:
                        break
                self.id =self.file.split(str('.'+ type_))[0]
                self.type = '.nii.gz'
            elif type_ == self.list_type_dicom[i]:
                self.id = self.file.split(str('.'+type_))[0]
                self.type = '.dcm'
                
    def ReadFile(self):
        self.GetInfo()
        if self.CheckTypeOfFile():
            return self.id, self.id + self.type
        return None, None
# if __name__ == '__main__':
#     os_path = os.getcwd()
#     data_path = 'D:/Spyder/Image/VU QUANG TAO M15'
#     database = {'ID':[], 'Path': []}
#     for file in os.listdir(data_path):
#         load_data = LoadData(file)
#         id_, path = load_data.ReadFile()
#         if id_ == None and path == None:
            