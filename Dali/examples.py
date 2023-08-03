import math
import os
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
#from nvidia.dali.backend import oss
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator
from nvidia.dali.types import DALIDataType

class CommonPipeline(Pipeline):
    def __init__(self,
                 batch_size,
                 num_workers,
                 image_size=(256, 256),
                 crop_size=(224, 224),
                 image_mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                 image_std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                 random_area=[0.08, 1.0],
                 random_aspect_ratio=[0.75, 1.333333],
                 train=True,
                 device_id=0,
                 shard_id=0,
                 seed=0,
                 decoder_device='mixed',
                 **kwargs):
        super(CommonPipeline, self).__init__(batch_size,
                                             num_workers,
                                             device_id,
                                             seed=seed + shard_id,
                                             **kwargs)
        self.train = train
        self.dali_device = 'gpu' if decoder_device == 'mixed' else 'cpu',
        self.decoder_device = decoder_device

        if train:
            device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
            host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
            self.decode = ops.ImageDecoderRandomCrop(
                device=decoder_device,
                output_type=types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding,
                random_area=random_area,
                random_aspect_ratio=random_aspect_ratio,
                num_attempts=100)
            self.coin = ops.CoinFlip(probability=0.5)
            self.resize = ops.Resize(
                                device='gpu' if decoder_device == 'mixed' else 'cpu',
                                resize_x=crop_size[1],
                                resize_y=crop_size[0],
                                interp_type=types.INTERP_TRIANGULAR)
        else:
            self.decode = ops.ImageDecoder(device=decoder_device,
                                           output_type=types.RGB)
            self.resize = ops.Resize(
                                device='gpu' if decoder_device == 'mixed' else 'cpu',
                                resize_x=image_size[1],
                                resize_y=image_size[0],
                                interp_type=types.INTERP_TRIANGULAR)

        assert isinstance(image_size, tuple) or isinstance(image_size, list)
        assert isinstance(crop_size, tuple) or isinstance(crop_size, list)
        assert len(image_size) == 2
        assert len(crop_size) == 2

        self.cnmp = ops.CropMirrorNormalize(
            device='gpu' if decoder_device == 'mixed' else 'cpu',
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop_w=crop_size[1],
            crop_h=crop_size[0],
            image_type=types.RGB,
            mean=image_mean,
            std=image_std)
        self.augmentations = []

    def base_define_graph(self, inputs, targets):
        inputs = self.decode(inputs)
        for augment in self.augmentations:
            inputs = augment(inputs)
        inputs = self.resize(inputs)
        if self.dali_device == 'gpu':
            inputs = inputs.gpu()
            targets = targets.gpu()
        if self.train:
            inputs = self.cnmp(inputs, mirror=self.coin())
        else:
            inputs = self.cnmp(inputs)
        return inputs, targets

    def define_graph(self):
        raise NotImplementedError

    def add_augmentations(self, ops):
        """Add augmentation list
        Args:
            ops: list of DALI ops that will perform image augmentations on decoded image data.
        """
        self.augmentations + ops

class ExternalSourcePipeline(CommonPipeline):
    def __init__(self, sampler_iterator,
                 batch_size,
                 num_workers,
                 image_size=(256, 256),
                 crop_size=(224, 224),
                 image_mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                 image_std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                 random_area=[0.08, 1.0],
                 random_aspect_ratio=[0.75, 1.333333],
                 train=True,
                 device_id=0,
                 shard_id=0,
                 seed=0,
                 decoder_device='mixed',
                 **kwargs):
        super(ExternalSourcePipeline, self).__init__(
                             batch_size, num_workers, image_size, crop_size,
                             image_mean, image_std, random_area,
                             random_aspect_ratio, train, device_id, shard_id,
                             seed, decoder_device, **kwargs)
        self.sampler_iterator = sampler_iterator
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        return self.base_define_graph(self.jpegs, self.labels)

    def iter_setup(self):
        (images, labels) = next(self.sampler_iterator)
        self.feed_input(self.jpegs, images)
        self.feed_input(self.labels, labels)

    def reset_sampler_iterator(self, sampler_iterator):
        self.sampler_iterator = sampler_iterator

class ClassificationIterator(DALIGenericIterator):
    def __init__(self,
                 sampler,
                 pipelines,
                 size,
                 fill_last_batch=True,
                 last_batch_padded=False):
        super(ClassificationIterator, self).__init__(pipelines, ["data", "label"],
                                                         size, auto_reset = False,
                                                         fill_last_batch = fill_last_batch,
                                                         dynamic_shape = False,
                                                         last_batch_padded = last_batch_padded)
        self.sampler = sampler
    
    def __len__(self):
        return ceil(self._size / self.batch_size)

    def reset(self, epoch):
        self.sampler.set_epoch(epoch)
        for p in self._pipes: # 在源码中self._pipes = pipelines
            p.reset_sampler_iterator(iter(self.sampler))
        super(ClassificationIterator, self).reset()

def make_dataloader(sampler, pipeline, num_shards, train):
    pipeline.build()
    print('pipeline build successful')
    assert len(sampler) % num_shards == 0
    size = len(sampler) / num_shards
    print('pipeline size{}'.format(size))
    if train:
        return ClassificationIterator(sampler=sampler, pipelines=pipeline,
                                                   size=size,
                                                   fill_last_batch=True,
                                                   last_batch_padded=False)
    else:
        return ClassificationIterator(sampler=sampler, pipelines=pipeline,
                                                   size=size,
                                                   fill_last_batch=False,
                                                   last_batch_padded=True)
                                             
import os
import time
import random
import numpy as np
import copy 
from io import StringIO

class Sampler(object):
    def __init__(self, data_root, file_list, batch_size, delimiter=','):
        self.data_root = data_root  #获取数据文件根目录
        self.batch_size = batch_size # 指定构造的batch的batch_size
        lines = open(file_list).readlines() # 读取文件
        # (代码的意思,file_list中写入了多个file的路径, 每个file表示一张图像)
        self.samples = [line.strip().split(delimiter) for line in lines if line is not ''] #每一行为一个file路径, self.samples为所有行组成的list

    def __iter__(self): # 采样组成batch
        batch = []
        labels = []
        for idx, sample in enumerate(self.samples):
            jpeg_filename, label = sample[0].split()
            f = open(self.data_root + jpeg_filename)
            batch.append(np.frombuffer(f.read(), dtype = np.uint8)) # 读取某一张图片数据,并加入batch
            labels.append(np.array([label], dtype = np.int32))
            if len(batch) == self.batch_size:
                yield (batch, labels)
                batch = []
                labels = []
        if len(batch) > 0: # 如果所有行都遍历完了, 还没达到batch_size, 就重复采样
            base_len = len(batch)
            for i in range(self.batch_size - base_len):
                img_obj = batch[i % base_len].copy()
                label_obj = labels[i % base_len].copy()
                batch.append(img_obj)
                labels.append(label_obj)
            yield (batch, labels)

    def __len__(self):
        return len(self.samples) #该函数获取数据的总数量. 目录文件中有多少行, 样本长度(总数量)就是多少.
"""
file_list.txt:
  0.jpg 0\n
  1.jpg 1\n
  ...
Note the 'xxx.jpg's have different sizes.
"""
sampler = Sampler(data_root="images/", file_list="images/file_list.txt", batch_size=4, delimiter=',')  
pipeline = ExternalSourcePipeline(sampler_iterator=iter(sampler),
                                   batch_size=4,
                                   num_workers=2,
                                   image_size=(256, 256),
                                   crop_size=(224, 224),
                                   image_mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                   image_std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                   random_area=[0.25, 1.0],
                                   random_aspect_ratio=[0.75, 1.333333],
                                   train=True,
                                   device_id=0,
                                   shard_id=1,
                                   decoder_device='mixed',
                                   prefetch_queue_depth=2,
                                   )
dataloader = make_dataloader(sampler, pipeline, 1, True)
for data in dataloader:
    image_data, label = data[0]["data"], data[0]["label"]
    print("ok")