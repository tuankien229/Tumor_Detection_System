# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:24:30 2022

@author: tuank
"""
# img_view.py
import PySimpleGUI as sg
import os.path
# from utils.Load import LoadImg, LoadData
import matplotlib.pyplot as plt
# First the window layout in 2 columns

file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-")
    ],
    [
        sg.Button('Show Image', enable_events=True,  key="-SHOW-")
    ]
]
# For now will only show the name of the file that was chosen
imag_viewer_column = [
    [sg.Text("Choose an image for list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(enable_events=True, key="-IMAGE-")],
]
# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(imag_viewer_column),
    ]
]
window = sg.Window("Tumor System", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in make a list of files in folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f for f in file_list
            if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((".nii.gz",".dcm", ".nii", ".DCM"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-SHOW-": # A file was chosen from the listbox
        try:
            # filename = os.path.join(
            #     values["-FOlDER-"], values["-FILE LIST-"][0]
            # )
            load_data = LoadData(values["-FOLDER-"])
            database = load_data.ReadFolder()
            load_image = LoadImg(database)
            image = load_image.StackImage().plot()
            window["-TOUT-"].update(values["-FOLDER-"])
            window["-IMAGE-"].update(data = image)
        except:
            pass
window.close()

