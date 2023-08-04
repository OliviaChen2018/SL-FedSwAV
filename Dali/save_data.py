# 按Dirichlet分布划分数据集, 并将划分后的index保存下来, 每次训练都使用同一种划分结果, 以便于复现.

from load_cifar10_data import *

image_dir = "../data/"
save_dir = "data/"

data, targets = load_cifar10(train=True, root=image_dir)
training_data_list, training_label_list, traindata_cls_counts, net_dataidx_map = partition_data(data,
                                                                               targets,
                                                                               num_client=10,
                                                                               num_class=10, 
                                                                               partition = 'noniid', 
                                                                               beta=0.4)
for client_index in range(len(training_data_list)): 
    np.save(save_dir+f"cifar{client_index}_index.npy", np.array(net_dataidx_map[client_index]))