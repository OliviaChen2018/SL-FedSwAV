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
import logging
from resnet import *

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,5,6"
local_rank = int(os.environ["LOCAL_RANK"]) # 使用这种方法获取local_rank,其他获取方法已经被弃用
print(f"local_rank: {local_rank}\n")
world_size = int(os.environ["WORLD_SIZE"])  #对于单机器, 可见设备有几张卡, world_size就是几.
print(f"world_size: {world_size}")

# 新增：DDP backend初始化
torch.cuda.set_device(local_rank) # 设置device
dist.init_process_group(backend='nccl')  # 初始化多进程. nccl是GPU设备上最快、最推荐的后端

#构造数据集
CIFAR10_MEAN=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.]
CIFAR10_STD=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]
IMG_DIR = '../data'
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

def setup_logger(name, log_file, level=logging.INFO, console_out = True):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    logger.addHandler(handler)
    if console_out:
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)
    return logger

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=36,kernel_size=3,stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        #self.aap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1296,128)
        self.fc2 = nn.Linear(128,10)
        #self.fc3 = nn.Linear(36,10)
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #x = self.aap(x)
        #x = x.view(x.shape[0],-1)
        #x = self.fc3(x)
        x = x.view(-1,36*6*6)
        #print("x.shape:{}".format(x.shape))
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return x
    
def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(world_size)]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]

data, targets = load_cifar10(train=True, root=image_dir)
test_data, test_targets = load_cifar10(train=False, root=image_dir)
# index = np.load("data/cifar0_index.npy")
model_log_file = output_dir + '/test.log'
logger = setup_logger('default_logger', model_log_file, level=logging.DEBUG)
sampler = CIFAR_INPUT_ITER(data, targets, batch_size, train=True, is_distributed=True)
test_sampler = CIFAR_INPUT_ITER(test_data, test_targets, batch_size, train=False)
pip_train = DaliTrainPipe_CIFAR(sampler, 
                                    data,
                                    targets,
                                    batch_size=batch_size, 
                                    num_threads=NUM_WORKERS, 
                                    device_id=local_rank, 
                                    world_size=world_size, 
                                    local_rank=local_rank)
pip_test = DaliTrainPipe_CIFAR(sampler, 
                                    test_data,
                                    test_targets,
                                    batch_size=batch_size, 
                                    num_threads=NUM_WORKERS, 
                                    device_id=local_rank, 
                                    world_size=world_size,
                                    local_rank=local_rank)
# pip_train = DaliTrainPipe_CIFAR_multicrop(sampler, 
#                                     data,
#                                     targets,
#                                     batch_size=batch_size, 
#                                     num_threads=NUM_WORKERS, 
#                                     device_id=local_rank, 
#                                     size_crops=[224,96], 
#                                     nmb_crops = [2,6], 
#                                     min_scale_crops=[0.14,0.05], 
#                                     max_scale_crops=[1,0.14],
#                                     world_size=world_size, 
#                                     local_rank=local_rank)
train_loader = DALIDataloader(sampler, 
                              pipeline=pip_train, 
                              size=sampler.len() / world_size,
                              batch_size=batch_size)
test_loader = DALIDataloader(test_sampler, 
                              pipeline=pip_test, 
                              size=test_sampler.len() / world_size,
                              batch_size=batch_size)

print(f"data of local_rank:{local_rank}")
print("[DALI] train dataloader length: %d"%len(train_loader))
logger.debug(f"data len of local_rank {local_rank}: {len(train_loader)}")
print('[DALI] start iterate train dataloader')

# 构造模型
device = torch.device("cuda", local_rank) # 获取device，之后的模型和张量都.to(device)
# model = ToyModel().to(local_rank)
# model = CNNNet().to(local_rank)
model = ResNet18().to(local_rank)
# model = nn.Linear(10, 10).to(device)
# 引入SyncBN，这句代码，会将普通BN替换成SyncBN。
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
# 新增：构造DDP model
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
# DDP: 要在构造DDP model之后，才能用model初始化optimizer。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# 假设我们的loss是这个
loss_func = nn.CrossEntropyLoss().to(local_rank)


### 3. 网络训练  ###
start = time.time()
for epoch in range(200):
    for i, data in enumerate(train_loader):
        images = data['data'].to(local_rank)
        labels = data['label'].t().squeeze().to(local_rank)
        optimizer.zero_grad()
        prediction = model(images)
        loss = loss_func(prediction, labels)
        loss.backward()
        optimizer.step()
        logger.debug(f"loss of {local_rank}: {loss}")
    train_loader.reset(epoch+1, num_shards = world_size, shard_id = local_rank, shuffle = True)
    train_loader._ever_consumed = False # 这一步是为了防止train_loader自动调用.reset()函数
    
    correct = 0
    total_num = 0
    with torch.no_grad():
        predictions = []
        labels = []
        for i, data in enumerate(test_loader):
            images = data['data'].to(local_rank)
            label = data['label'].t().squeeze().to(local_rank)
            predictions.append(model(images))
            labels.append(label)
        # 进行gather
        predictions = distributed_concat(torch.concat(predictions, dim=0), 
                                         len(test_data))
        labels = distributed_concat(torch.concat(labels, dim=0), 
                                    len(test_data))
        # 3. 现在我们已经拿到所有数据的predictioin结果，进行evaluate！
#         my_evaluate_func(predictions, labels)
        pre_labels= torch.argmax(predictions, dim=1)
#         print(f"size of prediction: {prediction.size()}")
        correct += (pre_labels==labels).sum().item()
        
        correct /= predictions.size(0)
        print(f"epoch:{epoch} acc is :{correct}")
        test_loader.reset(epoch=0)
        test_loader._ever_consumed = False # 这一步是为了防止train_loader自动调用.reset()函数

end = time.time()
train_time = end-start
print('[DALI] end train dataloader iteration')
print('[DALI] iteration time: %fs [train]' % (train_time))

# # 前向传播
# outputs = model(torch.randn(20, 10).to(device))
# labels = torch.randn(20, 10).to(device)
# loss_fn = nn.MSELoss()
# loss = loss_fn(outputs, labels)
# print(f"loss:{loss}")
# loss.backward()
# # 后向传播
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# optimizer.step()
