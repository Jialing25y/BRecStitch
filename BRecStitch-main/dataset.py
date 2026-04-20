from torch.utils.data import Dataset
import numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random


class TrainDataset(Dataset):
    def __init__(self, data_path):

        self.width = 512
        self.height = 512
        self.train_path = data_path
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input-1' or data_name == 'input-2' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                image_extensions = ['*.jpg', '*.JPG', '*.png', '*.PNG']
                self.datas[data_name]['image'] = []
                for ext in image_extensions:
                    self.datas[data_name]['image'].extend(glob.glob(os.path.join(data, ext)))
                self.datas[data_name]['image'].sort()
        
        print(self.datas.keys())

    def __getitem__(self, index):
        
        # load image1
        input1 = cv2.imread(self.datas['input-1']['image'][index])
        input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])
        
        # load image2
        input2 = cv2.imread(self.datas['input-2']['image'][index])
        input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])
        
        # convert to tensor
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)
        
        # randomly swap the order of images
        if_exchange = random.randint(0,1)
        if if_exchange == 0:
            return (input1_tensor, input2_tensor)
        else:
            return (input2_tensor, input1_tensor)

    def __len__(self):

        return len(self.datas['input-1']['image'])

class TestDataset(Dataset):
    def __init__(self, data_path):

        self.width = 512
        self.height = 512
        self.test_path = data_path
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input-1' or data_name == 'input-2' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                image_extensions = ['*.jpg', '*.JPG', '*.png', '*.PNG']
                self.datas[data_name]['image'] = []
                for ext in image_extensions:
                    self.datas[data_name]['image'].extend(glob.glob(os.path.join(data, ext)))
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())

    def __getitem__(self, index):
        
        # load image1
        input1 = cv2.imread(self.datas['input-1']['image'][index])
        input1 = cv2.resize(input1, (self.width, self.height))
        # input1 = cv2.resize(input1, (512, 512))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])
        
        # load image2
        input2 = cv2.imread(self.datas['input-2']['image'][index])
        input2 = cv2.resize(input2, (self.width, self.height))
        # input2 = cv2.resize(input2, (512, 512))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])
        
        # convert to tensor
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)

        return (input1_tensor, input2_tensor)

    def __len__(self):

        return len(self.datas['input-1']['image'])