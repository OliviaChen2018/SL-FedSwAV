# from nvidia.dali.pipeline import pipeline_def
# from nvidia.dali.pipeline import Pipeline
# import nvidia.dali.types as types
# import nvidia.dali.fn as fn
# from nvidia.dali.plugin.pytorch import DALIGenericIterator
# import nvidia.dali.ops as ops
import os
import sys
import pickle
import numpy as np
import pdb

# CIFAR10_MEAN=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.]
# CIFAR10_STD=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]
# IMG_DIR = '../data'
# CIFAR_IMAGES_NUM_TRAIN = 50000
# TRAIN_BS = 256
# NUM_WORKERS = 4
# CROP_SIZE = 32


#### 加载cifar10数据集 ####
def load_cifar10(batch_size, train=True, root='../data'):
    '''该函数返回的结果与torchvision.datasets.CIFAR10()函数取self.data和self.targets返回的结果相同'''
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    
    if train:
        downloaded_list = train_list
    else:
        downloaded_list = test_list

    data = []
    targets = []
    for file_name, checksum in downloaded_list:
        file_path = os.path.join(root, base_folder, file_name)
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            data.append(entry['data'])
            if 'labels' in entry:
                targets.extend(entry['labels'])
            else:
                targets.extend(entry['fine_labels'])

    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC
    targets = np.vstack(targets)
#     np.save("cifar.npy", data)
#     data = np.load('cifar.npy')  # to serialize, increase locality
    return data, targets


def partition_data(training_data, labels, num_client, num_class, partition = 'noniid', beta=0.4): 
    '''按照Dirichlet分布划分原始数据集 '''
    # 参数num_client表示client的数量
    # training_data和labels是numpy数组
    training_data_subset_list = []
    training_label_subset_list = []
#     pdb.set_trace()
    if partition == "homo" or partition == "iid":
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
            training_data_subset = training_data[idx_batch[j]]
            training_label_subset = labels[idx_batch[j]]
#             print(f"labels的维度为:{labels.shape}")
#             print(f"idx_batch[{j}]:{idx_batch[j]}")
            training_data_subset_list.append(training_data_subset)
            training_label_subset_list.append(training_label_subset)
#     print(net_dataidx_map)
    #traindata_cls_counts：数据分布情况（每个client拥有的所有类别及其数量）
    
    traindata_cls_counts = record_net_data_stats(labels, net_dataidx_map) 

    return training_data_subset_list, training_label_subset_list, traindata_cls_counts

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