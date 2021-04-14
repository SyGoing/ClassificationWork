#coding=utf-8
from torch.utils.data import Dataset
import cv2
import os
import torch
import numpy as np
from PIL import Image
class MaskDataSet(Dataset):
    def __init__(self,label_file,data_root,transform=None):
        self.data_root=data_root
        self.img_paths=[]
        self.labels=[]
        self.transforms=transform
        with open(label_file, 'r') as f:
            if data_root=="":
                for line in f.readlines():  # 读取label文件
                    self.img_paths.append(line.split()[0])
                    self.labels.append(int(line.split()[1]))
            else:
                for line in f.readlines():  # 读取label文件
                    self.img_paths.append(os.path.join(data_root, line.split()[0]))
                    self.labels.append(int(line.split()[1]))

    def __len__(self):
        return  len(self.img_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.img_paths[item])
        label=self.labels[item]
        if self.transforms is not None:
            return self.transforms(image),label
        else:
            return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1),label

#for mean teacher training
class MeanTeacherDataSet(Dataset):
    def __init__(self,label_file,data_root,transform=None,is_train=True):
        self.data_root = data_root
        self.img_paths = []
        self.labels = []
        self.transforms = transform
        self.isTrain=is_train
        with open(label_file, 'r') as f:
            if data_root == "":
                for line in f.readlines():  # 读取label文件
                    self.img_paths.append(line.split()[0])
                    self.labels.append(int(line.split()[1]))
            else:
                for line in f.readlines():  # 读取label文件
                    self.img_paths.append(os.path.join(data_root, line.split()[0]))
                    self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.img_paths[item])
        #image= Image.open(self.img_paths[item])
        label = self.labels[item]
        if self.transforms is not None:
            if self.isTrain:
                return self.transforms(image), self.transforms(image), label
            else:
                return self.transforms(image), label
