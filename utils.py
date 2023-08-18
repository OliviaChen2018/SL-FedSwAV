import torch
import numpy as np
import logging
import copy
import sys
from PIL import ImageFilter
import random
import torchvision
from torchvision.utils import make_grid
from torch.utils.data.distributed import DistributedSampler
# import matplotlib.pyplot as plt
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level=logging.INFO, console_out = True):
    """To setup as many loggers as you want"""

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

def average_weights(w, pool = None): 
    """
    Returns the average of the weights.
    （按层名计算列表w中的模型参数的均值，pool表示本次参与计算的model在w中的索引）
    """
    w_avg = copy.deepcopy(w[0].state_dict())
    for key in w_avg.keys(): # 按层计算均值
        if pool is None:
            for i in range(1, len(w)):
                w_avg[key] += w[i].state_dict()[key]
            w_avg[key] = torch.true_divide(w_avg[key], len(w))
        else:
            for i in range(1, len(pool)):
                w_avg[key] += w[pool[i]].state_dict()[key]
            w_avg[key] = torch.true_divide(w_avg[key], len(pool))
    return w_avg

def weighted_avg_weights(local_list, net_data_counts:dict, pool = None): 
    """
    Returns the average of the weights.
    （按层名计算列表w中的模型参数的均值，pool表示本次参与计算的model在w中的索引）
    """
    w_avg = copy.deepcopy(local_list[0].state_dict())
    for key in w_avg.keys(): # 按层计算均值
        if pool is None:
            for i in range(1, len(local_list)):
                w_avg[key] += local_list[i].state_dict()[key] * (net_data_counts[i] / sum(net_data_counts.values()))
        else:
            for i in range(1, len(pool)):
                w_avg[key] += local_list[pool[i]].state_dict()[key] * (net_data_counts[pool[i]] / sum(net_data_counts.values()))
    return w_avg

def accuracy(output, target, topk=(1,)): # 返回top1,top2,...,topk所有的正确率
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # pred的第一列表示预测出的最大概率值对应的class，第二列表示预测出的第二大概率值对应的class,...
    pred = pred.t() # 取转置，每行表示所有样本第k可能的class (第一行表示top1的class；第二行表示top2的class,...）
    correct = pred.eq(target.view(1, -1).expand_as(pred)) 
    # target.view(1, -1)将标签变为行向量；.expand_as(pred)扩展行(copy第一行到其他行)
    # pred.eq()计算每行中哪些与标签相同，相同则correct中该项为True，不同则为False。

    res = []
    for k in topk: # 分别计算topk的正确率。如k==0，则取correct中的前1行，计算所有true的个数；当k==1时，取correct中的前2行，并使用.view(-1)拉成一行，并计算所有true的个数，即只要前两行中存在与标签相同的预测值，则计入正确数，此时得到的结果即top2正确率。
        correct_k = correct[:k].view(-1).float().sum(0) 
        res.append(correct_k.mul_(100.0 / batch_size)) # res存放了top1,top2,...所有的正确率
    return res

def distributed_concat(tensor, num_total_examples, world_size):
    output_tensors = [tensor.clone() for _ in range(world_size)]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]

def sparsify(tensor, compress_ratio, num_grad):
    # num_grad表示tensor的第0维的维数，即tensor中有多少个grad.
    dim_1 = tensor.size(1) # 获取tensor的维数
    tensor = tensor.view(-1)
    samples = tensor.abs() #对tensor中的元素取绝对值
    numel = tensor.numel()
    top_k = max(1, int(numel * compress_ratio))
    threshold = torch.min(torch.topk(samples, top_k, dim=0, largest=True, sorted=False)[0]) # 找出前k个最大值中的最小值，作为阈值
    mask = torch.ge(samples, threshold) # 设置掩码，大于阈值的位置为1，小于阈值的位置为0
    indices = mask.nonzero().view(-1) # 得到选中元素的下标
    tensor = tensor[indices]
    return tensor, indices #问题: 这个tensor是压缩的, 怎么用来反向传播?
    

class AverageMeter(object): # 一个用于存储和 和 均值的对象
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1): 
        # 假设有n个数，每个数的值都为val，计算n个值的均值，并保存在self.avg中
        # 由于类对象的变量sum保存了和，因此eval的时候可以对不同batch的acc和进行累加
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        images, labels = self.dataset[self.idxs[item]]
        return images, labels
    
def partition_data(training_data, labels, num_client, shuffle, num_workers, batch_size, num_class, partition = 'noniid', beta=0.4, is_distributed=False): 
     #参数num_client表示client的数量
    if num_client == 1:
        if is_distributed:
            train_sampler = DistributedSampler(training_subset)
            shuffle = False
        else:
            train_sampler=None
        training_loader_list = [torch.utils.data.DataLoader(training_data,  batch_size=batch_size, shuffle = shuffle, sampler = train_sampler, num_workers=num_workers, pin_memory=True)]
        
    elif num_client > 1:
        training_loader_list = []
        if partition == "homo" or partition == "iid":
            N = labels.shape[0]
            idxs = np.random.permutation(N)   #在训练集的条数范围内生成随机序列
            batch_idxs = np.array_split(idxs, num_client)
            net_dataidx_map = {i: batch_idxs[i] for i in range(num_client)}

        elif partition == "noniid-labeldir" or partition == "noniid":
            min_size = 0 
            min_require_size = 10   # 每个client至少要有10条数据
            K = num_class
                # min_require_size = 100

            N = labels.shape[0]
            net_dataidx_map = {}  #用于存放每个client拥有的样本的idx数组

            #min_size表示所有client中样本数量最少的client对应的样本数量。如果存在某个client的样本数量没达到min_require_size，则继续为client分配样本。
            while min_size < min_require_size:
                idx_batch = [[] for _ in range(num_client)]  # idx_batch存放num_client个client对应的样本idx
                for k in range(K): #遍历所有类别，将每个类别按Dirichlet分布的比例分配给各个client。
                    idx_k = np.where(labels == k)[0]  #idx_k表示训练集中label为k的所有样本的idx集合
                    np.random.shuffle(idx_k) #上面选出来的idx是按顺序的，现在把顺序打乱。
                    proportions = np.random.dirichlet(np.repeat(beta, num_client)) 
                    #proportions的长度为num_client
                    proportions = np.array([p * (len(idx_j) < N / num_client) for p, idx_j in zip(proportions, idx_batch)])  # 取出第j个client拥有的所有sample下标和第j个client的idx
                    proportions = proportions / proportions.sum() #将剩下的client的划分比例重新归一化
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1] #
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]  #为第j个client分配类别k的样本
                    min_size = min([len(idx_j) for idx_j in idx_batch]) #min_size表示所有client中样本数量最少的client对应的样本数量
                    # if K == 2 and num_client <= 10:
                    #     if np.min(proportions) < 200:
                    #         min_size = 0
                    #         break

            
            for j in range(num_client):
                #分配完之后，由于idx_batch中的样本idx是按类别顺序存放的，所以要打乱。
                np.random.shuffle(idx_batch[j]) 
                net_dataidx_map[j] = idx_batch[j] # 用net_dataidx_map记录每个client拥有的样本。
                # 封装为dataloader
                training_subset = torch.utils.data.Subset(training_data, idx_batch[j])
                if is_distributed:
                    train_sampler = DistributedSampler(training_subset)
                    shuffle = False
                else:
                    train_sampler = None
                if num_workers > 0:
                    subset_training_loader = torch.utils.data.DataLoader(training_subset,
                                                                         shuffle=shuffle, 
                                                                         num_workers=num_workers,
                                                                         batch_size=batch_size, 
                                                                         sampler = train_sampler,
                                                                         persistent_workers = True,
                                                                         pin_memory=True)
                else:
                    subset_training_loader = torch.utils.data.DataLoader(training_subset, 
                                                                         shuffle=shuffle, 
                                                                         num_workers=num_workers, 
                                                                         batch_size=batch_size, 
                                                                         sampler = train_sampler,
                                                                         persistent_workers = False,
                                                                         pin_memory=True)
                training_loader_list.append(subset_training_loader)
#     print(net_dataidx_map)
    #traindata_cls_counts：数据分布情况（每个client拥有的所有类别及其数量）
    traindata_cls_counts = record_net_data_stats(labels, net_dataidx_map) 

    return training_loader_list, traindata_cls_counts

def record_net_data_stats(y_train, net_dataidx_map):
    '''用于记录每个client的数据分布(拥有的所有样本类别，及该类别出现的次数)'''
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items(): # dict.items()返回(key, value)元组组成的列表
    # net_i表示第i个client, dataidx为其拥有的样本idx
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True) #返回unique的类别数组
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))} # 字典,存放第i个client拥有的类别及其数量
        net_cls_counts[net_i] = tmp #字典，存放所有client的类别信息

    data_list=[]
    for net_id, data in net_cls_counts.items(): # net_id表示client编号，data表示该client拥有的类别的次数信息
        n_total=0
        for class_id, n_data in data.items(): # class_id表示类别编号，n_data表示该类别在该client中的出现次数
            n_total += n_data  # 计算该client拥有的数据条数
        data_list.append(n_total) #data_list保存每个client拥有的数据条数
    print('mean:', np.mean(data_list)) #打印每个client的平均数据条数和方差，以显示异质程度
    print('std:', np.std(data_list))

    return net_cls_counts


def get_multiclient_trainloader_list(training_data, num_client, shuffle, num_workers, batch_size, noniid_ratio = 1.0, num_class = 10, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", is_distributed=False):
    #mearning of default hetero_string = "C_D|B" - dividing clients into two groups, stronger group: C clients has D of the data (batch size = B); weaker group: the other (1-C) clients have (1-D) of the data (batch size = 1).
    net_data_counts = {i:0 for i in range(num_client)} #记录每个client拥有的数据量
    '''返回所有client的dataloader组成的list'''
    if num_client == 1: # 只有1个client的时候
        # 对于mem_loader和test_loader，num_client为默认值1
        if is_distributed:
            train_sampler = DistributedSampler(training_subset)
            shuffle = False
        else:
            train_sampler=None
        training_loader_list = [torch.utils.data.DataLoader(training_data, 
                                                            batch_size=batch_size, 
                                                            shuffle=shuffle,
                                                            sampler = train_sampler,
                                                            num_workers=num_workers)]
        net_data_counts[0] = len(training_data)
    elif num_client > 1:
        if noniid_ratio < 1.0:
            training_subset_list = noniid_alllabel(training_data, num_client, noniid_ratio, num_class, hetero, hetero_string) # TODO: implement non_iid_hetero version.
        
        training_loader_list = []
        
        # rich_client指拥有数据较多的client; rich_data指rich_client所拥有的数据量；
        if hetero: 
            rich_data_ratio = float(hetero_string.split("|")[-1].split("_")[0]) # rich_data_ratio==0.8
            rich_data_volume = int(rich_data_ratio * len(training_data)) # 用于分配的rich_data的数量
            rich_client_ratio = float(hetero_string.split("|")[0].split("_")[0]) # rich_client_ratio==0.2
            rich_client = int(rich_client_ratio * num_client) # rich_client的数量

        for i in range(num_client):
            # print(f"client {i}:")
            if noniid_ratio == 1.0:
                if not hetero:
                    training_subset = torch.utils.data.Subset(training_data, list(range(i * (len(training_data)//num_client), (i+1) * (len(training_data)//num_client))))
                    net_data_counts[i] = len(training_data)//num_client
                else:
                    if i < rich_client:
                        training_subset = torch.utils.data.Subset(training_data, list(range(i * (rich_data_volume//rich_client), (i+1) * (rich_data_volume//rich_client))))
                        # 为每个rich_client分配rich_data_volume//rich_client条数据
                        net_data_counts[i] = rich_data_volume//rich_client
                    elif i >= rich_client: # 为非rich_client分配数据
                        heteor_list = list(range(rich_data_volume + (i - rich_client) * (len(training_data) - rich_data_volume) // (num_client - rich_client), rich_data_volume + (i - rich_client + 1) * (len(training_data) - rich_data_volume) // (num_client - rich_client)))
                        net_data_counts[i] = (len(training_data) - rich_data_volume) // (num_client - rich_client)
                        training_subset = torch.utils.data.Subset(training_data, heteor_list)
                
            else:
                training_subset = DatasetSplit(training_data, training_subset_list[i])
                net_data_counts[i] = len(list(training_subset_list[i]))
            # print(len(training_subset))
            if is_distributed:
                train_sampler = DistributedSampler(training_subset)
                shuffle = False
            else:
                train_sampler=None
            if not hetero:
                if num_workers > 0:
                    subset_training_loader = torch.utils.data.DataLoader(
                        training_subset, shuffle=shuffle, sampler = train_sampler, num_workers=num_workers, batch_size=batch_size, persistent_workers = True)
                else:
                    subset_training_loader = torch.utils.data.DataLoader(
                        training_subset, shuffle=shuffle, sampler = train_sampler, num_workers=num_workers, batch_size=batch_size, persistent_workers = False)
            else:
                if i < rich_client:
                    real_batch_size = batch_size * int(hetero_string.split("|")[1]) 
                    # 为什么rich_client的batch_size要多乘16？
                elif i >= rich_client:
                    real_batch_size = batch_size
                #当有多个workers时，设置persistent_workers为True，使workers一直开启。
                if num_workers > 0: 
                    subset_training_loader = torch.utils.data.DataLoader(
                        training_subset, shuffle=shuffle, sampler = train_sampler, num_workers=num_workers, batch_size=real_batch_size, persistent_workers = True)
                else:
                    subset_training_loader = torch.utils.data.DataLoader(
                        training_subset, shuffle=shuffle, sampler = train_sampler, num_workers=num_workers, batch_size=real_batch_size, persistent_workers = False)
                # print(f"batch size is {real_batch_size}")
            training_loader_list.append(subset_training_loader)
    
    return training_loader_list, net_data_counts


class Subset(torch.utils.data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def noniid_unlabel(dataset, num_users, label_rate, noniid_ratio = 0.2, num_class = 10):
    #计算noniid_ratio比例下每个client拥有的类别数量
    num_class_per_client = int(noniid_ratio * num_class) 
    num_shards, num_imgs = num_class_per_client * num_users, int(len(dataset)/num_users/num_class_per_client)
    # num_shards表示所有clients的类别数量，num_imgs表示每个client分到的img的数量
    idx_shard = [i for i in range(num_shards)]
    #dict_users_unlabeled用于存放所有client分配到的样本的下标idx
    dict_users_unlabeled = {i: np.array([], dtype='int64') for i in range(num_users)} 
    idxs = np.arange(len(dataset)) # 用于给原始样本添加下标
    labels = np.arange(len(dataset))  #用于存放每条数据的labels
    

    for i in range(len(dataset)):
        labels[i] = dataset[i][1] #第i条数据的[1]，即第i条数据的标签
        
    dict_users_labeled = set()

    # sort labels 将原始按labels数据排序，以便后面按类别进行分配
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_class_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_unlabeled[i] = np.concatenate((dict_users_unlabeled[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0) # 将对应的样本(下标)分配给第i个client

    dict_users_labeled = set(np.random.choice(list(idxs), int(len(idxs) * label_rate), replace=False))
    
    for i in range(num_users):

        dict_users_unlabeled[i] = set(dict_users_unlabeled[i])
        dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled


    return dict_users_labeled, dict_users_unlabeled


# def visualize_classification(loader_iter, labelMap = None, nrofItems = 16, pad = 4, save_name = "unknown"):

#   #Iterate through the data loader
#   imgTensor, labels = next(loader_iter)
  
#   # Generate image grid
#   grid = make_grid(imgTensor[:nrofItems], padding = pad, nrow=nrofItems)

#   # Permute the axis as numpy expects image of shape (H x W x C) 
#   grid = grid.permute(1, 2, 0)

#   # Get Labels
#   if labelMap is not None:
#       labels = [labelMap[lbl.item()] for lbl in labels[:nrofItems]]
#   else:
#       labels = [f"unknown" for lbl in labels[:nrofItems]]
#   # Set up plot config
#   plt.figure(figsize=(8, 2), dpi=300)
#   plt.axis('off')

#   # Plot Image Grid
#   plt.imshow(grid)
  
#   # Plot the image titles
#   fact = 1 + (nrofItems)/100
#   rng = np.linspace(1/(fact*nrofItems), 1 - 1/(fact*nrofItems) , num = nrofItems)
#   for idx, val in enumerate(rng):
#     plt.figtext(val, 0.85, labels[idx], fontsize=8)

#   # Show the plot
# #   plt.show()
#   plt.savefig(f"visual{save_name}.png")


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def divergence_plot(path_to_log, freq = 1):
    file1 = open(path_to_log, 'r')
    Lines = file1.readlines()
    
    count = 0
    divergence_mean_list = []
    divergence_std_list = []
    # Strips the newline character
    divergence_mean = 0
    divergence_std = 0
    for line in Lines:
        
        if "divergence mean:" in line:
            count += 1
            divergence_mean += float(line.split("divergence mean: ")[-1].split(", std:")[0])
            divergence_std += float(line.split(", std: ")[-1].split(" and detailed_list:")[0])
            
            if count % freq == 0:
                
                divergence_mean_list.append(divergence_mean/freq)
                divergence_std_list.append(divergence_std/freq)
                divergence_mean = 0
                divergence_std = 0
                count = 0
    
    return divergence_mean_list, divergence_std_list




def noniid_alllabel(dataset, num_users, noniid_ratio = 0.2, num_class = 10, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2"):
    num_class_per_client = int(noniid_ratio * num_class)
    # 500 clients -> *5 = 2500 clients.
    if hetero:
        num_shards_multiplier = float(hetero_string.split("|")[-1].split("_")[-1]) # 0.2 (last float)
        num_shards = int(num_class_per_client  * num_users  / num_shards_multiplier) # more shards (equivalent to more clients)
        num_imgs = int(len(dataset)/num_users/num_class_per_client * num_shards_multiplier) # less image
        rich_client_ratio = float(hetero_string.split("|")[0].split("_")[0]) # 0.2 (first float)
        rich_client = int(rich_client_ratio * num_users) # 100 clients
        rich_client_gets_shards = int((1-num_shards_multiplier)/num_shards_multiplier) # each get 4 shards
    else:
        num_shards, num_imgs = num_class_per_client * num_users, int(len(dataset)/num_users/num_class_per_client)
    # print(f"num_shards: {num_shards}, num_imgs: {num_imgs}")

    idx_shard = [i for i in range(num_shards)]
    dict_users_labeled = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.arange(len(dataset))
    for i in range(len(dataset)):
        if dataset.__class__.__name__ == "Subset":
            labels[i] = dataset.dataset.targets[dataset.indices[i]] #dataset must be a subset
        else:
            labels[i] = dataset[i][1]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    if not hetero:
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, num_class_per_client, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users_labeled[i] = np.concatenate((dict_users_labeled[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    else:
        virtual_num_user = rich_client * rich_client_gets_shards + num_users - rich_client
        for i in range(virtual_num_user):
            rand_set = set(np.random.choice(idx_shard, num_class_per_client, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            if i < rich_client * rich_client_gets_shards: # assign shards for rich clients
                for rand in rand_set:
                    dict_users_labeled[i // rich_client_gets_shards] = np.concatenate((dict_users_labeled[i // rich_client_gets_shards], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
            else:
                for rand in rand_set:
                    dict_users_labeled[(i - rich_client * rich_client_gets_shards) + rich_client] = np.concatenate((dict_users_labeled[(i - rich_client * rich_client_gets_shards) + rich_client], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)


    for i in range(num_users):
        # print(f"user {i} has {len(dict_users_labeled[i])} images")
        dict_users_labeled[i] = set(dict_users_labeled[i])

    return dict_users_labeled

class CustomSubset(Subset):
    '''A custom subset class'''
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
#         self.targets = dataset.targets # 保留targets属性
#         self.classes = dataset.classes # 保留classes属性
        
        if torchvision.__version__ == '0.2.1':
            self.target = dataset.train_labels
#             data, self.target = train_data.train_data, np.array(train_data.train_labels) 
            #torchvision.datasets.CIFAR10官方类自己会处理train_data或test_data。
        else:
#             data = train_data.data
            self.targets = dataset.targets
        # data表示数据集中的所有样本值，target表示样本标签。

    def __getitem__(self, idx): #同时支持索引访问操作
        x, y = self.dataset[self.indices[idx]]      
        return x, y 

    def __len__(self): # 同时支持取长度操作
        return len(self.indices)



if __name__ == '__main__':
    #avgfreq
    avg_freq = 1
    cutlayer = 3
    file_name = f'mocosflV2_ResNet18_cifar10_cut{cutlayer}_bnlNone_client5_nonIID0.2_avg_freq_{avg_freq}'
    # file_name = f'mocosflV2_ResNet18_cifar10_cut{cutlayer}_bnlNone_client5_nonIID0.2'
    path_to_log = f'outputs/divergence/{file_name}/output.log'


    file_name = 'mocofl_ResNet18-cifar10_crosssilo_batchsize128_nonIID0.2_client5_subsample_1.0_local_epoch_5'
    path_to_log = f'outputs/{file_name}/output.log'


    divergence_mean_list, divergence_std_list = divergence_plot(path_to_log, avg_freq)
    print(divergence_mean_list)

    #cutlayer
    # avg_freq = 1
    # cutlayer = 4
    # file_name = f'mocosflV2_ResNet18_cifar10_cut{cutlayer}_bnlNone_client5_nonIID0.2'
    # path_to_log = f'outputs/divergence/{file_name}/output.log'
    # divergence_mean_list, divergence_std_list = divergence_plot(path_to_log, avg_freq)
    # print(divergence_mean_list)