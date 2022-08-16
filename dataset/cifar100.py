from __future__ import print_function

import os
import socket
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    
    #data_folder = '/data/home/xdeng7/code/EnhancedKD/cifar100/data'
    data_folder = './data'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder




def get_cifar100_dataloaders(batch_size=64, num_workers=8, is_shuffle=True):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    
    train_set = datasets.CIFAR100(root=data_folder,
                                  download=True,
                                  train=True, 
                                  transform=train_transform)
    
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=is_shuffle,
                              num_workers=num_workers)




    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))
    
    return train_loader, test_loader


class EnCIFAR100(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    """

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, ensamples=None):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        
        self.ensamples = ensamples

    def __getitem__(self, index):
        if self.train:
            #img, target = self.train_data[index], self.train_labels[index]
            img, target, enimg = self.data[index], self.targets[index], self.ensamples[index]
        else:
            img, target, enimg = self.test_data[index], self.test_labels[index], self.ensamples[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        enimg = Image.fromarray(enimg)

        if self.transform is not None:
            img = self.transform(img)
            enimg = self.transform(enimg)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, enimg

class TraValCIFAR100(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    """

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, ratio = 0.9):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        
        self.ratio = ratio
        
        idx = np.random.RandomState(seed=1).permutation(len(self.data))
        
        #print('targer shape', self.data.shape)
        self.targets = np.array(self.targets)
        
        shuffle_data,  shuffle_targets = self.data[idx], self.targets[idx]
        
        tran_num = int(len(self.data)*ratio)
        
        
        self.data, self.targets = shuffle_data[ 0: tran_num], shuffle_targets[ 0:tran_num ]
        
        self.val_data, self.val_targets = shuffle_data[ tran_num: ], shuffle_targets[ tran_num: ]
        

    def __getitem__(self, index):
        
        
        if self.train:
            
            index2 = index%len(self.val_data)
            
            #img, target = self.train_data[index], self.train_labels[index]
            img, target, valimg, val_labels = self.data[index], self.targets[index], self.val_data[index2], self.val_targets[index2]
        else:
            img, target, valimg, val_labels = self.test_data[index], self.test_labels[index], None, None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        valimg = Image.fromarray(valimg)

        if self.transform is not None:
            img = self.transform(img)
            valimg = self.transform(valimg)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, valimg, val_labels


