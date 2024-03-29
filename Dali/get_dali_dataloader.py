from Dali.load_cifar10_data import load_cifar10, partition_data, record_net_data_stats
from Dali.Dali_Dataloader import DALIDataloader
from Dali.cifar_Dali_Dataset import DaliTrainPipe_CIFAR_multicrop, DaliTrainPipe_CIFAR, CIFAR_INPUT_ITER
import torch
import numpy as np
import pdb

def get_cifar10_Dali_loader(train_sampler_list, training_data_list, training_label_list, num_client, batch_size, data_portion = 1.0, num_workers = 4, device_id=0, dali_cpu=False, local_rank=0, world_size=1, cutout=0, train = True):
    '''普通的cifar10的dali读取
    train_loader和test_loader都用这个,以传入的data和targets的不同自动区分.
    返回包含dataloader的list'''
    assert len(training_data_list)==num_client # 数据集划分的个数与client的数量要保持一致
    assert len(train_sampler_list)==num_client
    
    loader_list = []
    
    if num_client == 1: # 只有1个client的时候
        # 对于mem_loader和test_loader，num_client为默认值1
        train_sampler = train_sampler_list[0]
        train_data = training_data_list[0] #获取当前client的数据
        train_label = training_label_list[0]

        pip_train = DaliTrainPipe_CIFAR(train_sampler, train_data, train_label, 
                                      batch_size=batch_size, 
                                      num_threads=num_workers, 
                                      device_id=device_id,   
                                      world_size=world_size, 
                                      local_rank=local_rank, 
                                      cutout=cutout,
                                      train = train)
        subset_loader = DALIDataloader(train_sampler,
                                       pipeline=pip_train, 
                                       size=train_sampler.len() / world_size,
                                       batch_size=batch_size)  #创建当前client的dataloader
        loader_list.append(subset_loader) 
    elif num_client > 1:
        for i in range(num_client):
            train_sampler = train_sampler_list[i]
            train_data = training_data_list[i] #获取当前client的数据
            train_label = training_label_list[i]
            # 创建数据集pipeline
            pip_train = DaliTrainPipe_CIFAR(train_sampler,
                                                      train_data,
                                                      train_label,
                                                      batch_size=batch_size, 
                                                      num_threads=num_workers, 
                                                      device_id=device_id,
                                                      world_size=world_size, 
                                                      local_rank=local_rank, 
                                                      cutout=cutout,
                                                      train = train)
            subset_loader = DALIDataloader(train_sampler,
                                           pipeline=pip_train, 
                                           size=train_sampler.len() / world_size,
                                           batch_size=batch_size)  #创建当前client的dataloader
            loader_list.append(subset_loader) 
    return loader_list


def get_cifar10_Dali_multicroploader(train_sampler_list, training_data_list, training_label_list, num_client, batch_size, num_workers = 4, nmb_crops = [2], size_crops=[224], min_scale_crops=[0.14], max_scale_crops=[1], device_id=0, dali_cpu=False, local_rank=0, world_size=1, cutout=0, train = True):
    '''多个clients的数据执行multicrop数据增强.返回所有client的dataloader组成的list'''
    assert len(training_data_list)==num_client # 数据集划分的个数与client的数量要保持一致
    assert len(train_sampler_list)==num_client
    # 创建数据集
    training_loader_list = []
    if num_client == 1: # 只有1个client的时候
        # 对于mem_loader和test_loader，num_client为默认值1
        train_sampler = train_sampler_list[0]
        train_data = training_data_list[0] #获取当前client的数据
        train_label = training_label_list[0]
        pip_train = DaliTrainPipe_CIFAR_multicrop(train_sampler,
                                                      train_data,
                                                      train_label,
                                                      batch_size=batch_size, 
                                                      num_threads=num_workers, 
                                                      device_id=device_id,
                                                      nmb_crops = nmb_crops,
                                                      size_crops=size_crops, 
                                                      min_scale_crops=min_scale_crops,
                                                      max_scale_crops=max_scale_crops, 
                                                      world_size=world_size, 
                                                      local_rank=local_rank, 
                                                      cutout=cutout,
                                                      train = train)
        subset_training_loader = DALIDataloader(train_sampler,
                                                pipeline=pip_train, 
                                                size=train_data.shape[0],
                                                batch_size=batch_size)  #创建当前client的dataloader
        training_loader_list.append(subset_training_loader) 
        
    elif num_client > 1:
        for i in range(num_client):
            train_sampler = train_sampler_list[i]
            train_data = training_data_list[i] #获取当前client的数据
            train_label = training_label_list[i]
            # 创建数据集pipeline
#             pdb.set_trace()
            pip_train = DaliTrainPipe_CIFAR_multicrop(train_sampler,
                                                      train_data,
                                                      train_label,
                                                      batch_size=batch_size, 
                                                      num_threads=num_workers, 
                                                      device_id=device_id,
                                                      nmb_crops = nmb_crops,
                                                      size_crops=size_crops, 
                                                      min_scale_crops=min_scale_crops,
                                                      max_scale_crops=max_scale_crops, 
                                                      world_size=world_size, 
                                                      local_rank=local_rank, 
                                                      cutout=cutout,
                                                      train = train)
            subset_training_loader = DALIDataloader(train_sampler,
                                                    pipeline=pip_train, 
                                                    size=train_data.shape[0],
                                                    batch_size=batch_size)  #创建当前client的dataloader
            training_loader_list.append(subset_training_loader) 
    
    return training_loader_list


def get_cifar10_dali(size_crops=None, nmb_crops=None, min_scale_crops=None, max_scale_crops=None, batch_size=16, num_workers=4, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, pairloader_option = "None", partition = 'noniid', partition_beta = 0.4, hetero = False, path_to_data = "../data"):
    # 先读取cifar10数据
    train_data, train_targets = load_cifar10(train=True, root=path_to_data)
    test_data, test_targets = load_cifar10(train=False, root=path_to_data)
    
    if data_proportion < 1.0 and data_proportion > 0.0: #这个采样操作是用于mem_loader的
#         indices = torch.randperm(len(train_data))[:int(len(train_data)* data_portion)]
        indices = np.random.permutation(len(train_data))[:int(len(train_data)* data_portion)]
        train_data = train_data[indices]
        train_targets = train_targets[indices]
        
    training_data_list, training_label_list, traindata_cls_counts = partition_data(train_data,
                                                                               train_targets,
                                                                               num_client=num_client,
                                                                               num_class=10, 
                                                                               partition = partition, 
                                                                               beta=partition_beta)
    train_sampler_list = []
    for client_index in range(num_client):
        train_sampler = CIFAR_INPUT_ITER(training_data_list[client_index], training_label_list[client_index], batch_size, train=True)
        train_sampler_list.append(train_sampler)
        
    if pairloader_option != "None": #要做数据增强
        # train_loader用于contrastive fl的训练, 是一个包含所有client train_dataloader的list；
        # test_loader用于validate
        # mem_loader也是一个包含所有client dataloader的list；
        if data_proportion <= 0.0:
            train_loader = None
        else: # pairloader也并在这里面一起,与multicrop通过size_crops中的元素的值和个数自动区分
#             pdb.set_trace()
            train_loader = get_cifar10_Dali_multicroploader(
                train_sampler_list, training_data_list, training_label_list,
                num_client, batch_size, num_workers, 
                nmb_crops, size_crops, min_scale_crops, max_scale_crops, train=True)
#         pdb.set_trace()
        mem_sampler = CIFAR_INPUT_ITER(train_data, train_targets, batch_size=128, train=True)
        mem_loader = get_cifar10_Dali_loader([mem_sampler], [train_data], [train_targets], num_client = 1, batch_size=128, num_workers = num_workers, train=False)
        
        test_sampler = CIFAR_INPUT_ITER(test_data, test_targets, batch_size=128, train=False)
        test_loader = get_cifar10_Dali_loader([test_sampler], [test_data], [test_targets], num_client=1, batch_size=128, num_workers = num_workers, train=False)
        
        return train_loader, traindata_cls_counts, mem_loader, test_loader
    
    else: # 不做数据增强
        if data_proportion > 0.0:
            train_loader = get_cifar10_Dali_loader(train_sampler_list, training_data_list, training_label_list, num_client, batch_size, num_workers, train=False)
        else:
            train_loader = None
            
        test_sampler = CIFAR_INPUT_ITER(test_data, test_targets, batch_size=128, train=False)
        test_loader = get_cifar10_Dali_testloader([test_sampler], [test_data], [test_targets], num_client=1, batch_size = 128, num_workers = num_workers, train=False)

        return train_loader, traindata_cls_counts, test_loader

    
def get_cifar10_dali_DDP(size_crops=None, nmb_crops=None, min_scale_crops=None, max_scale_crops=None, batch_size=16, num_workers=4, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, pairloader_option = "None", partition = 'noniid', partition_beta = 0.4, hetero = False, path_to_data = "../data", local_rank = 0, world_size = 1):
    # 先读取cifar10数据(在此之前先运行save_data.py进行数据划分)
    train_data, train_targets = load_cifar10(train=True, root=path_to_data)
    test_data, test_targets = load_cifar10(train=False, root=path_to_data)
#     if data_proportion < 1.0 and data_proportion > 0.0:
# #         indices = torch.randperm(len(train_data))[:int(len(train_data)* data_portion)]
#         indices = np.random.permutation(len(train_data))[:int(len(train_data)* data_portion)]
#         train_data = train_data[indices]
#         train_targets = train_targets[indices]
    net_dataidx_map = {i:None for i in range(num_client)}
    if pairloader_option != "None": #要做数据增强
        # train_loader用于contrastive fl的训练, 是一个包含所有client train_dataloader的list；
        # test_loader用于validate
        # mem_loader也是一个包含所有client dataloader的list；
        if data_proportion <= 0.0:
            train_loader = None
        else: # pairloader也并在这里面一起,与multicrop通过size_crops中的元素的值和个数自动区分
            train_loader = []
            for client_index in range(num_client):
                data_index = np.load(f"data/cifar{client_index}_index.npy")
                net_dataidx_map[client_index] = data_index
                sampler = CIFAR_INPUT_ITER(train_data[data_index], train_targets[data_index], batch_size, train = True, world_size = world_size, local_rank = local_rank, is_distributed=True)
                pip_train = DaliTrainPipe_CIFAR_multicrop(sampler,
                                                              train_data[data_index],
                                                              train_targets[data_index],
                                                              batch_size=batch_size, 
                                                              num_threads=num_workers, 
                                                              device_id=0,
                                                              nmb_crops = nmb_crops,
                                                              size_crops=size_crops, 
                                                              min_scale_crops=min_scale_crops,
                                                              max_scale_crops=max_scale_crops, 
                                                              world_size=world_size, 
                                                              local_rank=local_rank,
                                                              train = True)
                dataloader = DALIDataloader(sampler, 
                                          pipeline=pip_train, 
                                          size=sampler.len() / world_size,
                                          batch_size=batch_size)
                train_loader.append(dataloader)
        
        mem_sampler = CIFAR_INPUT_ITER(train_data, train_targets, batch_size=128, train=True, world_size = world_size, local_rank = local_rank, is_distributed = True)
        mem_loader = get_cifar10_Dali_loader([mem_sampler], [train_data], [train_targets], num_client = 1, batch_size=128, num_workers = num_workers, train=False, world_size = world_size, local_rank = local_rank)
        
        test_sampler = CIFAR_INPUT_ITER(test_data, test_targets, batch_size=128, train=False, world_size = world_size, local_rank = local_rank, is_distributed = True)
        test_loader = get_cifar10_Dali_loader([test_sampler], [test_data], [test_targets], num_client=1, batch_size=128, num_workers = num_workers, train=False, world_size = world_size, local_rank = local_rank)
        
        traindata_cls_counts = record_net_data_stats(train_targets, net_dataidx_map)
        return train_loader, traindata_cls_counts, mem_loader, test_loader
    
    else: # 不做数据增强
        if data_proportion > 0.0:
            train_sample_list = []
            training_data_list = []
            training_label_list = []
            for client_index in range(args.num_client):
                data_index = np.load(f"data/cifar{client_index}_index.npy")
                net_dataidx_map[client_index] = data_index
                sampler = CIFAR_INPUT_ITER(train_data[data_index], train_targets[data_index], batch_size, train = True, world_size = world_size, local_rank = local_rank, is_distributed=True)
                train_sample_list.append(sampler)
                training_data_list.append(train_data[data_index])
                training_label_list.append(train_targets[data_index])
            train_loader = get_cifar10_Dali_loader(train_sample_list, training_data_list, training_label_list, num_client, batch_size, num_workers, train=False, world_size = world_size, local_rank = local_rank)
        else:
            train_loader = None
            
        test_sampler = CIFAR_INPUT_ITER(test_data, test_targets, batch_size=128, train=False, world_size = world_size, local_rank = local_rank, is_distributed = True)
        test_loader = get_cifar10_Dali_testloader([test_sampler], [test_data], [test_targets], num_client=1, batch_size = 128, num_workers = num_workers, train=False, world_size = world_size, local_rank = local_rank)

        return train_loader, traindata_cls_counts, test_loader