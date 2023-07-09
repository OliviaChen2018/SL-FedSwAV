import numpy as np
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader
from utils import GaussianBlur, partition_data
from PIL import Image
import os
import os.path
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def get_cifar10(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, augmentation_option = False, pairloader_option = "None", partition = 'noniid', hetero = False, path_to_data = "./data"):
    if pairloader_option != "None": # moco, swav需要做数据增强，需要构造pairloader
        train_loader = get_cifar10_pairloader(batch_size, num_workers, shuffle, num_client, data_portion=data_proportion, pairloader_option, partition, hetero = False, path_to_data)
        mem_loader = get_cifar10_trainloader(128, num_workers, False, path_to_data = path_to_data)
        test_loader = get_cifar10_testloader(128, num_workers, False, path_to_data)
        return train_loader, mem_loader, test_loader
    else: # 传统FL不需要做数据增强，不需要构造pairloader
        train_loader = get_cifar10_trainloader(batch_size, num_workers, shuffle, num_client, data_portion=data_proportion, augmentation_option, partition, path_to_data)
        test_loader = get_cifar10_testloader(128, num_workers, False, path_to_data)
        return train_loader, test_loader


def get_cifar10_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, augmentation_option = False, partition, hetero = False, path_to_data = "./data"):
    """ return training dataloader
    Args:
        mean: mean of cifar10 training dataset
        std: std of cifar10 training dataset
        path: path to cifar10 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    if augmentation_option:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
        ])
    #cifar00_training = CIFAR10Train(path, transform=transform_train)
    cifar10_training = datasets.CIFAR10(root=path_to_data, train=True, download=True, transform=transform_train)
    
    # 只取原始CIFAR10训练集的一部分作为训练集？
    indices = torch.randperm(len(cifar10_training))[:int(len(cifar10_training)* data_portion)]

    cifar10_training = torch.utils.data.Subset(cifar10_training, indices)
    
    if torchvision.__version__ == '0.2.1':
        data, target = cifar10_training.train_data, np.array(cifar10_training.train_labels) 
        #torchvision.datasets.CIFAR10官方类自己会处理train_data或test_data。
    else:
        data = cifar10_training.data
        target = np.array(cifar10_training.targets)
    # data表示数据集中的所有样本值，target表示样本标签。

    cifar10_training_loader = partition_data(cifar10_training, target, num_client, shuffle, num_workers, batch_size, num_class=10, partition = partition, hetero ,beta=0.4)

    return cifar10_training_loader


def get_cifar10_pairloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, pairloader_option = "None", partition = 'noniid', hetero = False, path_to_data = "./data"):
    class CIFAR10Pair(torchvision.datasets.CIFAR10):
        """CIFAR10 Dataset.
        """
        def __getitem__(self, index):
            img = self.data[index]
            img = Image.fromarray(img)
            if self.transform is not None:
                im_1 = self.transform(img)
                im_2 = self.transform(img)
            return im_1, im_2
        
    if pairloader_option == "mocov1":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
    elif pairloader_option == "mocov2":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
    elif pairloader_option=='swav':
        pass
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
        
    # data prepare
    train_data = CIFAR10Pair(root=path_to_data, train=True, transform=train_transform, download=True) #train_data中包含若干个img pair元组
    
    indices = torch.randperm(len(train_data))[:int(len(train_data)* data_portion)]

    train_data = torch.utils.data.Subset(train_data, indices)
    
    if torchvision.__version__ == '0.2.1':
        data, target = train_data.train_data, np.array(train_data.train_labels) 
        #torchvision.datasets.CIFAR10官方类自己会处理train_data或test_data。
    else:
        data = train_data.data
        target = np.array(train_data.targets)
    # data表示数据集中的所有样本值，target表示样本标签。
    
    cifar10_training_loader = partition_data(train_data, target, num_client, shuffle, num_workers, batch_size, num_class=10, partition = partition, beta=0.4)
    
    return cifar10_training_loader