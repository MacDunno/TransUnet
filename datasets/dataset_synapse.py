import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
from PIL import Image 

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label'] 
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class RandomGenerator1(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label'] 
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class Synapse_dataset(Dataset):
    def __init__(self, img_dir, list_dir, split, label_dir=None, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.img_dir = img_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            img_path = os.path.join(self.img_dir, slice_name+'.png')
            # 使用 PIL 加载图像  
            image = Image.open(img_path).convert('L')  
            # 转换为 NumPy 数组  
            image = np.array(image)
            label_path = os.path.join(self.label_dir, slice_name+'.png')
            # 使用 PIL 加载图像  
            label = Image.open(label_path)   
            # 转换为 NumPy 数组  
            label = np.array(label)
            label[label == 255] = 1 
            image, label = image, label
            # if image.ndim == 3:  # 如果是多通道图像
            #     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif self.split == "test_only":
            slice_name = self.sample_list[idx].strip('\n')
            img_path = os.path.join(self.img_dir, slice_name+'.png')
            # 使用 PIL 加载图像  
            image = Image.open(img_path).convert('L')   
            # 转换为 NumPy 数组  
            image = np.array(image) 
        else:
            slice_name = self.sample_list[idx].strip('\n')
            img_path = os.path.join(self.img_dir, slice_name+'.png')
            image = Image.open(img_path).convert('L')  
            # 转换为 NumPy 数组  
            image = np.array(image)
            label_path = os.path.join(self.label_dir, slice_name+'.png')
            # 使用 PIL 加载图像  
            label = Image.open(label_path)    
            # 转换为 NumPy 数组  
            label = np.array(label)
            label[label == 255] = 1 
            image, label = image, label
        if self.split == "test_only":
            sample = {'image': image}
        else:
            sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
