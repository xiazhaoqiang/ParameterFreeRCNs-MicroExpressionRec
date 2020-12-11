import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class MEGC2019(torch.utils.data.Dataset):
    """MEGC2019 dataset class with 3 categories"""

    def __init__(self, imgList, transform=None):
        self.imgPath = []
        self.label = []
        self.dbtype = []
        with open(imgList,'r') as f:
            for textline in f:
                texts= textline.strip('\n').split(' ')
                self.imgPath.append(texts[0])
                self.label.append(int(texts[1]))
                self.dbtype.append(int(texts[2]))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open("".join(self.imgPath[idx]),'r').convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label[idx]

    def __len__(self):
        return len(self.imgPath)

class MEGC2019_SI(torch.utils.data.Dataset):
    """MEGC2019_SI dataset class with 3 categories and other side information"""

    def __init__(self, imgList, transform=None):
        self.imgPath = []
        self.label = []
        self.dbtype = []
        with open(imgList,'r') as f:
            for textline in f:
                texts= textline.strip('\n').split(' ')
                self.imgPath.append(texts[0])
                self.label.append(int(texts[1]))
                self.dbtype.append(int(texts[2]))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open("".join(self.imgPath[idx]),'r').convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
        return {"data":img, "class_label":self.label[idx], 'db_label':self.dbtype[idx]}

    def __len__(self):
        return len(self.imgPath)

class MEGC2019_FOLDER(torch.utils.data.Dataset):
    """MEGC2019 dataset class with 3 categories, organized in folders"""

    def __init__(self, rootDir, transform=None):
        labels = os.listdir(rootDir)
        labels.sort()
        self.fileList = []
        self.label = []
        self.imgPath = []
        for subfolder in labels:
            label = []
            imgPath = []
            files = os.listdir(os.path.join(rootDir, subfolder))
            files.sort()
            self.fileList.extend(files)
            label = [int(subfolder) for file in files]
            imgPath = [os.path.join(rootDir, subfolder,file) for file in files]
            self.label.extend(label)
            self.imgPath.extend(imgPath)
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.imgPath[idx],'r').convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
        return {"data":img, "class_label":self.label[idx]}

    def __len__(self):
        return len(self.fileList)