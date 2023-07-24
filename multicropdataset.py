# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger

from PIL import ImageFilter
from PIL import Image
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

logger = getLogger()

STL10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
STL10_TRAIN_STD = (0.2471, 0.2435, 0.2616)
CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
TINYIMAGENET_TRAIN_MEAN = (0.5141, 0.5775, 0.3985)
TINYIMAGENET_TRAIN_STD = (0.2927, 0.2570, 0.1434)
SVHN_TRAIN_MEAN = (0.3522, 0.4004, 0.4463)
SVHN_TRAIN_STD = (0.1189, 0.1377, 0.1784)
IMAGENET_TRAIN_MEAN = (0.485, 0.456, 0.406)
IMAGENET_TRAIN_STD = (0.229, 0.224, 0.225)


class MultiCropCifar10Dataset(datasets.CIFAR10):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        return_index=False,
        train=True,
        download=True
    ):
        super(MultiCropCifar10Dataset, self).__init__(root=data_path, train=train, 
                                                      download=download)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = CIFAR10_TRAIN_MEAN
        std = CIFAR10_TRAIN_STD
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
#         print(f"增强样本的个数：{len(trans)}")  # expected: 8
        self.trans = trans

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        multi_crops = list(map(lambda trans: trans(img), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class MultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
    ):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s) #改变图像的属性：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8) # 以0.8的概率随机执行list[color_jitter]中的变换 (Dali中没有RandomApply这个操作)
    rnd_gray = transforms.RandomGrayscale(p=0.2) #以0.2的概率进行随机灰度变换
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray]) #将颜色变换和灰度变换组合成一组transforms
    return color_distort