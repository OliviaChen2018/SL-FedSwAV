from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.pipeline import Pipeline
import nvidia.dali as dali
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import os
import sys
import pickle
import numpy as np
import cupy as cp
from sklearn.utils import shuffle
import pdb
from torch.utils.data import IterableDataset
from Dali.load_cifar10_data import *
from Dali.Dali_Dataloader import DALIDataloader
import time

CIFAR10_MEAN=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.]
CIFAR10_STD=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]

class CIFAR_INPUT_ITER():
    def __init__(self, data, targets, batch_size, train=True):
        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.train = train
        np.save("cifar.npy", self.data)
        self.data = np.load('cifar.npy')  # to serialize, increase locality #这一步一定要用,否则会报错???
        print(self.data.shape)

    def __iter__(self):
        self.i = 0
        self.n = len(self.data)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            if self.train and self.i % self.n == 0: # 训练集才shuffle
                print("执行shuffle")
                self.data, self.targets = shuffle(self.data, self.targets, random_state=0)
            img, label = self.data[self.i], self.targets[self.i]
            batch.append(img)
            labels.append(label)
            self.i = (self.i + 1) % self.n
#         pdb.set_trace()
        return (batch, labels)

    next = __next__
    
class DaliTrainPipe_CIFAR_multicrop(Pipeline): # 这个pipeline就相当于是在构造数据集
    def __init__(self, data, targets, batch_size, num_threads = 4, device_id=0, size_crops=[224], nmb_crops = [2], min_scale_crops=[0.14], max_scale_crops=[1], dali_cpu=False, local_rank=0, world_size=1, cutout=0, train=True):
        # Pipeline初始化函数的output_ndim参数: 指定pipeline返回的output list中每个output的维数.比如define_graph返回[output, labels], 则output_ndim应该设定为[output的维数, labels的维数]
        super(DaliTrainPipe_CIFAR_multicrop, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        self.iterator = iter(CIFAR_INPUT_ITER(data, targets, batch_size, train))
        dali_device = "gpu"
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.pad = ops.Paste(device=dali_device, ratio=1.25, fill_value=0)
        self.uniform = ops.random.Uniform(range=(0., 1.))
#         self.crop = ops.Crop(device=dali_device, crop_h=crop, crop_w=crop)
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            dtype=types.FLOAT,
                                            mean=CIFAR10_MEAN,
                                            std=CIFAR10_STD
                                            ) #这个函数可以实现镜面对称
        # image_type=types.RGB,
        self.coin = ops.random.CoinFlip(probability=0.5)
        self.sat = ops.random.Uniform(range=[0.2, 1.8]) # 亮度,饱和度,对比度变换的变化范围
        self.hue = ops.random.Uniform(range=[-0.2, 0.2]) # 色调 设置hue=0.2
        # 上面这三个值作为参数,其device必须是cpu,不然会报错.
        self.rrc_1 = ops.RandomResizedCrop(device=dali_device, size=size_crops[0], random_area =[min_scale_crops[0], max_scale_crops[0]] )
        if len(size_crops)==2:
            self.rrc_2 = ops.RandomResizedCrop(device=dali_device,
                                               size=size_crops[1],
                                               random_area=[min_scale_crops[1], max_scale_crops[1]] )
#         self.flip = ops.Flip(device=dali_device, horizontal=self.coin())
        rng = self.coin() #随机生成一个boolean值
        sat = self.sat() # 亮度,饱和度,对比度变换的变化范围
        hue = self.hue() # 色调 设置hue=0.2
        self.ColorJitter = ops.ColorTwist(device=dali_device)
        self.hsv = self.random_grayscale(probability=0.2)
#         self.hsv = ops.Hsv(device = dali_device, saturation=self.saturate)
        self.gaussian = ops.GaussianBlur(device=dali_device, sigma=[1.0, 2.0])
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = ops.RandomResizedCrop(device=dali_device,
                size=size_crops[i],
                random_area=(min_scale_crops[i], max_scale_crops[i])
            )
            trans.extend([ops.Compose([
                randomresizedcrop,
                ops.ColorTwist(device=dali_device, 
                                brightness = sat, saturation = sat, contrast=sat, hue=hue),
                self.random_grayscale(probability=0.2), 
                ops.GaussianBlur(device=dali_device, sigma=[1.0, 2.0]),
                ops.CropMirrorNormalize(device=dali_device,
                                            dtype=types.FLOAT,
                                            mean=CIFAR10_MEAN,
                                            std=CIFAR10_STD)])
            ] * nmb_crops[i])
        #image_type=types.RGB,
#         print(f"增强样本的个数：{len(trans)}")  # expected: 8
        self.trans = trans

    def iter_setup(self):
        (images, labels) = self.iterator.next() # 调用CIFAR_INPUT_ITER中的__next__函数
#         pdb.set_trace()
        self.feed_input(self.images, images, layout="HWC") # images是个list,包含batch_size条图片数据, 每条数据是ndarray类型,shape为[32,32,3].
        self.feed_input(self.labels, labels)
        

    def define_graph(self): #这个函数是定义怎么对图像进行变换.
        rng = self.coin() #随机生成一个boolean值
        sat = self.sat() # 亮度,饱和度,对比度变换的变化范围
        hue = self.hue() # 色调 设置hue=0.2
        self.images = self.input()
        self.labels = self.input_label()
        output = self.images
#         output1 = self.rrc_1(output.gpu())
#         output2 = self.rrc_2(output.gpu())
#         output = self.ColorJitter(output1, brightness = sat, saturation = sat, contrast=sat, hue=hue)
#         output = self.hsv(output)
#         output = self.gaussian(output)
#         output = self.cmnp(output, mirror=rng) #这里的rng是个随机boolean值,所以会随机镜面翻转
        multi_crops = list(map(lambda trans: trans(output.gpu()), self.trans))
        
        for i in range(len(multi_crops)-1): # 先把所有增强样本的size维度reshape拉成1行, 然后按列进行拼接后返回. 在获取batch之后先按列还原成若干个增强样本的输出, 再分别reshape为size的形状.
            if i == 0:
                output0 = fn.reshape(multi_crops[i], shape = [3, 1,-1])
                output1 = fn.reshape(multi_crops[i+1], shape = [3,1, -1])
                output = fn.cat(output0, output1, axis = 2)
            else:
                output1 = fn.reshape(multi_crops[i+1], shape = [3, 1,-1])
                output = fn.cat(output, output1, axis = 2)
        print(len(multi_crops))
#         return [*multi_crops, self.labels]
#         pdb.set_trace()
        return [output, self.labels]
    
    def random_grayscale(self, probability):
        saturate = ops.random.CoinFlip(probability=1-probability)
        saturate = ops.Cast(dtype=types.FLOAT)(saturate())
        hsv = ops.Hsv(device = "gpu", saturation=saturate)
        return hsv
    
    
class DaliTrainPipe_CIFAR(Pipeline): # 这个pipeline就相当于是在构造数据集
    def __init__(self, data, targets, batch_size, num_threads = 4, device_id=0, dali_cpu=False, local_rank=0, world_size=1, cutout=0, train=True):
#         pdb.set_trace()
        super(DaliTrainPipe_CIFAR, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.iterator = iter(CIFAR_INPUT_ITER(data, targets, batch_size, train))
        dali_device = "gpu"
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            dtype=types.FLOAT,
                                            image_type=types.RGB,
                                            mean=CIFAR10_MEAN,
                                            std=CIFAR10_STD) #这个函数可以实现镜面对称
    def iter_setup(self):
        (images, labels) = self.iterator.next() # 调用CIFAR_INPUT_ITER中的__next__函数
#         pdb.set_trace()
        self.feed_input(self.images, images, layout="HWC") # images是个list,包含batch_size条图片数据, 每条数据是ndarray类型,shape为[32,32,3].
        self.feed_input(self.labels, labels)
        

    def define_graph(self): #这个函数是定义怎么对图像进行变换.
        self.images = self.input()
        self.labels = self.input_label()
        output = self.images
#         output = self.cmnp(output, mirror=0) 
        output = self.cmnp(output.gpu()) # mirror参数默认为0
        return [output, self.labels]

