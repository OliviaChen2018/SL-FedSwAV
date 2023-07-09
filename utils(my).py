

def partition_data(training_data, labels, num_client, shuffle, num_workers, batch_size, num_class, partition = 'noniid', beta=0.4): 
     #参数num_client表示client的数量
    if num_client == 1:
        training_loader_list = [torch.utils.data.DataLoader(training_data,  batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)]

    elif num_client > 1:
        training_loader_list = []
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
#                 net_dataidx_map[j] = idx_batch[j] # 用net_dataidx_map记录每个client拥有的样本。
                # 封装为dataloader
                training_subset = torch.utils.data.Subset(training_data, idx_batch[j])
                if num_workers > 0:
                    subset_training_loader = torch.utils.data.DataLoader(training_subset,
                                                                         shuffle=shuffle, 
                                                                         num_workers=num_workers,
                                                                         batch_size=real_batch_size, 
                                                                         persistent_workers = True)
                else:
                    subset_training_loader = torch.utils.data.DataLoader(training_subset, 
                                                                         shuffle=shuffle, 
                                                                         num_workers=num_workers, 
                                                                         batch_size=real_batch_size, 
                                                                         persistent_workers = False)
                training_loader_list.append(subset_training_loader)
#         #traindata_cls_counts：数据分布情况（每个client拥有的所有类别及其数量）
#     traindata_cls_counts = cls.record_net_data_stats(labels, net_dataidx_map) 

    return training_loader_list