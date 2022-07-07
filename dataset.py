'''
Description: 
Author: Wu Yubo
Date: 2022-05-13 17:16:51
LastEditTime: 2022-05-19 21:00:29
LastEditors:  
'''
# -*- coding: utf-8 -*-

'''
@Time    : 13/5/2022 4:55 下午
@Author  : bobo
@FileName: dataset.py
@Software: PyCharm
 
'''
import numpy as np
import torch.utils.data
import mat4py


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, aug=False, input_channels = 1):
        self.args = args
        self.img_paths = img_paths
        self.aug = aug
        self.input_channels =input_channels
        print(self.input_channels)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        matData = mat4py.loadmat(img_path)
        npimage = np.array(matData['data'])
        npimage=(npimage - np.mean(npimage)) / np.std(npimage)

        nplabel = int(matData['label'])

        if self.input_channels > 0:
            npimage2 = np.zeros((1,npimage.shape[-1],npimage.shape[-2]))
            npimage2[0] = npimage
            npimage = npimage2

        #nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")
        

        return npimage,nplabel


        #读图片（如jpg、png）的代码
        '''
        image = imread(img_path)
        mask = imread(mask_path)

        image = image.astype('float32') / 255
        mask = mask.astype('float32') / 255

        if self.aug:
            if random.uniform(0, 1) > 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if random.uniform(0, 1) > 0.5:
                image = image[::-1, :, :].copy()
                mask = mask[::-1, :].copy()

        image = color.gray2rgb(image)
        #image = image[:,:,np.newaxis]
        image = image.transpose((2, 0, 1))
        mask = mask[:,:,np.newaxis]
        mask = mask.transpose((2, 0, 1))       
        return image, mask
        '''

