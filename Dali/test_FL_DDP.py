## main.py文件
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import argparse
import os
import torch.nn as nn
from load_cifar10_data import *
from cifar_Dali_Dataset import *
import numpy as np
import pdb
from utils import setup_logger, distributed_concat
from resnet import *
import logging

CIFAR10_MEAN=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.]
CIFAR10_STD=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]
CIFAR_IMAGES_NUM_TRAIN = 50000
NUM_WORKERS = 4
CROP_SIZE = 32

image_dir = "../data/"
output_dir = "dali_output/"
batch_size = 128
# NMB_CROPS = [2,6]
# SIZE_CROPS = [224,96]
# MIN_SCALE = [0.14,0.05]
# MAX_SCALE = [1,0.14]

model_log_file = output_dir + '/test_FL_DDP.log'
logger = setup_logger('default_logger', model_log_file, level=logging.DEBUG)

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,5,6"
local_rank = int(os.environ["LOCAL_RANK"]) # 使用这种方法获取local_rank,其他获取方法已经被弃用
print(f"local_rank: {local_rank}\n")
world_size = int(os.environ["WORLD_SIZE"])  #对于单机器, 可见设备有几张卡, world_size就是几.
print(f"world_size: {world_size}")

# 新增：DDP backend初始化
torch.cuda.set_device(local_rank) # 设置device
dist.init_process_group(backend='nccl')  # 初始化多进程. nccl是GPU设备上最快、最推荐的后端

#构造数据集
train_data, train_targets = load_cifar10(train=True, root=image_dir)
test_data, test_targets = load_cifar10(train=False, root=image_dir)

num_clients = 10
data_index_list = []
# sampler_list = []
# pipeline_list = []
train_loader = []
for client_index in range(num_clients):
    data_index = np.load(f"data/cifar{client_index}_index.npy")
    data_index_list.append(data_index)
    sampler = CIFAR_INPUT_ITER(train_data[data_index], train_targets[data_index], batch_size, train=True, world_size = world_size, local_rank = local_rank, is_distributed=True)
    pip_train = DaliTrainPipe_CIFAR(sampler, 
                                    train_data[data_index],
                                    train_targets[data_index],
                                    batch_size=batch_size, 
                                    num_threads=NUM_WORKERS, 
                                    device_id=local_rank, 
                                    world_size=world_size, 
                                    local_rank=local_rank)
    dataloader = DALIDataloader(sampler, 
                              pipeline=pip_train, 
                              size=sampler.len() / world_size,
                              batch_size=batch_size)
    train_loader.append(dataloader)

num_batch = len(train_loader[0])
for epoch in range(2):
    for client_index in range(num_clients):
        logger.debug(f"data length of client{client_index} in epoch{epoch}: {len(train_loader[client_index])}")
        for i, data in enumerate(train_loader[client_index]):
            pass
    for client_index in range(num_clients):
        train_loader[client_index].reset(epoch+1, num_shards = world_size, shard_id = local_rank, shuffle = True)
        train_loader[client_index]._ever_consumed = False # 这一步是为了防止train_loader自动调用.reset()函数
        