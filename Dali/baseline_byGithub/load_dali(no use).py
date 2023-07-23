from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.ops as ops
import os
import sys
import pickle
import numpy as np
import cupy as cp
from sklearn.utils import shuffle
import pdb
from torch.utils.data import IterableDataset
from load_cifar10_data import *

# class DALIDataset(IterableDataset):
#     def __init__(self, data, targets, batch_size, train= True, base_path=None, **kwargs):
#         super().__init__()
#         self.data = data
#         self.targets = targets
#         self.i = 0
#         self.n = len(self.data)
#         self.train=train
#         self.batch_size = batch_size

#     def __iter__(self):
#         batch = []
#         labels = []
#         for _ in range(self.batch_size):
#             if self.train and self.i % self.n == 0:
#                 print("执行shuffle")
# #                 pdb.set_trace()
#                 self.data, self.targets = shuffle(self.data, self.targets, random_state=0)
#             img, label = self.data[self.i], self.targets[self.i]
#             yield img, label
#             batch.append(img)
#             labels.append(label)
#             self.i = (self.i + 1) % self.n
#         return batch, labels

class DALIDataset(IterableDataset):
    def __init__(self, batch_size, base_path=None, **kwargs):
        super().__init__()
#         self.files = os.scandir(base_path)
        self.data, self.targets = load_cifar10(batch_size)
        self.i = 0
        self.n = len(self.data)
        
        def __iter__(self):
            if self.train and self.i % self.n == 0:
                img, label = self.data[self.i], self.targets[self.i]
                yield image, label
        

#Using ExternalSource
class SimplePipeline(Pipeline):
    #Define the operations in the pipeline
    def __init__(self, external_datasource, batch_size=16, num_threads=2, device_id=0, resolution=256, crop=224, is_train=True):
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        # Define Input nodes
        self.jpegs = ops.ExternalSource()
        self.in_labels = ops.ExternalSource()
        ## Or pass source straight to ExternalSource this way you won't have do iter_setup.
#         self.jpegs,self.labels=ops.ExternalSource(source=self.make_batch, num_outputs=2)
        
        # Define ops
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_x=resolution,resize_y=resolution)
        self.normalize= ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.path_pad=ops.Pad(fill_value=ord("?"),axes = (0,)) # We need to pad image_paths because we need the shapes to match.need dense tensor
        
        self.iterator = iter(external_datasource)

    # The external source should be fed batches
    # I prefer to batch-ify things here because it keeps things compatible with an IterableDataset
    def make_batch(self):
        imgs = []
        labels = []
        for _ in range(self.batch_size):
            i,l=next(self.iterator)
            imgs.append(i)
            labels.append(l)
        return (imgs,labels)
    
    # How the operations in the pipeline are used
    # Connect your input nodes to your ops
    def define_graph(self):
        self.images = self.jpegs()
        self.labels = self.in_labels()
        images = self.decode(self.images)
        images = self.res(images)
        images = self.normalize(images)
        return (images, self.labels)

    # Only needed when using ExternalSource
    # Connect the dataset outputs to external Sources
    def iter_setup(self):
        (images,labels) = self.make_batch()
        self.feed_input(self.images, images)
        self.feed_input(self.labels, labels)
        
def make_pipeline(dataset, args, device_index=0, return_keys=["images","labels"], num_threads=2, is_train=True):
    pipeline = SimplePipeline(dataset, 
                              batch_size=args["batch_size"], 
                              num_threads=num_threads,
                              device_id=device_index, 
                              resolution=args["resolution"], 
                              crop=args["crop"], is_train=is_train)
    pipeline_iterator = DALIGenericIterator(pipeline, return_keys)
    return pipeline_iterator


args = {
    "resolution": 256,
    "crop":224,
    "batch_size": 128,
    "image_folder": "../data/" # Change this
}
# data, targets = load_cifar10(batch_size, train=True, root=image_dir)
data, targets = load_cifar10(batch_size=128, train=True, root='../data')
training_data_list, training_label_list, traindata_cls_counts = partition_data(data,
                                                                               targets,
                                                                               num_client=10,
                                                                               num_class=10, 
                                                                               partition = 'noniid', 
                                                                               beta=0.4)
dataset = DALIDataset(batch_size = 256)
train_dataloader=make_pipeline(dataset,args)
for batch in train_dataloader:
    print(batch[0]["images"].shape,batch[0]["labels"].shape,batch[0]["image_path"].shape) 
    print(batch[0]["images"].device,batch[0]["labels"].device,batch[0]["image_path"].device)