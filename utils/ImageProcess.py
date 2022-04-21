# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:47:43 2022

@author: tuank
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
class ImageProcess():
    # def Threshold(self, image):
        # image, _, _ = self.CheckShape(image)
        # plt.subplot(121)
        # _, thresh_1 = cv2.threshold(image,50, 255, cv2.THRESH_OTSU)
        # plt.imshow(thresh_1)
        # plt.subplot(122)
        # _, thresh_2 = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
        # plt.imshow(thresh_2)
        # plt.show()
        # return thresh_1
    def RemoveThresh(self, seg, type_):
        _, thresh = cv2.threshold(seg, 0, 255, type_)
        removed = np.where(thresh, self.image, 0)
        return removed
    def CheckShape(self, image):
        if len(image.shape) == 3:
            image = image.squeeze(2)
        wight, hight = image.shape
        return image, wight, hight
            
    def RemoveNoise(self, image):
        self.image, _, _ = self.CheckShape(image)
        _, lis_seg = self.K_means(self.image, 5)
        noise_removed = self.RemoveThresh(lis_seg[0], cv2.THRESH_BINARY_INV)
        noise_removed = cv2.fastNlMeansDenoising(noise_removed, 10, 10, 7, 21)
        # _, lis_seg = self.K_means(self.image, 5)
        # noise_removed = self.RemoveThresh(lis_seg[0], cv2.THRESH_BINARY_INV)
        # plt.imshow(noise_removed)
        # plt.show()
        return noise_removed
        
    def RemoveRice(self, image):
        self.image, _, _ = self.CheckShape(image)
        _, thresh_image = cv2.threshold(self.image, 10, 255, cv2.THRESH_OTSU)
        output  = cv2.connectedComponentsWithStats(thresh_image, 8, cv2.CV_32S)
        numLabels, labels, stats, centroids = output
        mask = np.zeros(self.image.shape, np.uint8)
        area_max = 0
        w_max = 0
        h_max = 0
        x_max = 0
        y_max = 0
        
        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if area > area_max:
                area_max = area
                w_max = w
                h_max = h
                x_max = x
                y_max = y
        # for i in range(1, numLabels):
        #     x = stats[i, cv2.CC_STAT_LEFT]
        #     y = stats[i, cv2.CC_STAT_TOP]
        #     w = stats[i, cv2.CC_STAT_WIDTH]
        #     h = stats[i, cv2.CC_STAT_HEIGHT]
        #     area = stats[i, cv2.CC_STAT_AREA]
        #     if x >= x_max and y >= y_max and x + w <= x_max + w_max and y + h <= y_max + h_max:
        #     # if w > 20 and h > 20:
        #         componentMask = (labels == i).astype("uint8") * 255
        #         mask = cv2.bitwise_or(mask, componentMask)
        mask[y_max: y_max + h_max, x_max: x_max + w_max] = self.image[y_max: y_max + h_max, x_max: x_max + w_max]
        # rice_removed = np.where(mask, self.image, 0)
        rice_removed = mask
        # plt.imshow(mask)
        # plt.show()
        return rice_removed
    def RemoveSkull(self, root_file):
        INPUT_FOLDER = os.path.join(root_file + '\nii_file', 'brain_dcm.nii.gz')
        OUTPUT_FOLDER = root_file + '\nii_file'
        os.chdir(f"{os.path.join(os.getcwd(), 'HD-BET')}")
        print(INPUT_FOLDER,OUTPUT_FOLDER, os.getcwd())
        if torch.cuda.is_available():
            if torch.cuda.mem_get_info()[1]/1e9 > 6:
                command = f'hd-bet/hd-bet -i {INPUT_FOLDER} -o {OUTPUT_FOLDER}'
            else:
                command = f'hd-bet/hd-bet -i {INPUT_FOLDER} -o {OUTPUT_FOLDER} -device cpu -mode fast -tta 0'
        else:
            command = f'hd-bet/hd-bet -i {INPUT_FOLDER} -o {OUTPUT_FOLDER} -device cpu -mode fast -tta 0'
        print(command)
        os.system(command)
        os.chdir(root_file)
        
    def K_means(self, im_gray, K):
        img_copy = im_gray.copy()
        img_copy = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2RGB)
        pixels = img_copy.reshape((-1,3))
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.001)
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        result = centers[labels.flatten()]
        img_seg = result.reshape(img_copy.shape)
        img_seg = cv2.cvtColor(img_seg, cv2.COLOR_RGB2GRAY)
        lis_cen = []
        for cen in centers:
            lis_cen.append(cen[0])
        lis_cen = np.sort(lis_cen)
        lis_seg = []
        for c in range(len(lis_cen)):
            img_test = np.zeros(img_seg.shape, np.uint8)
            for x in range(img_seg.shape[0]):
                for y in range(img_seg.shape[1]):
                    if img_seg[x,y] == lis_cen[c]:
                        img_test[x, y] = lis_cen[c]
            lis_seg.append(img_test)
        return img_seg, lis_seg