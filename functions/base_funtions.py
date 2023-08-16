'''
SFL basic functionality, wrap in a class. 

Can be extended to real-applcations if communication protocol is considered.

To understand how the training works in this implementation of SFL. We provide a tutorial in __main__ function:

Refer to Thapa et al. https://arxiv.org/abs/2004.12088 for technical details.

'''

from email.policy import strict
import torch
import logging
from utils import AverageMeter, accuracy, average_weights, setup_logger, SinkhornDistance
import pdb

class base_simulator:
    def __init__(self, model, criterion, train_loader, test_loader, device, args) -> None:
        if not model.cloud_classifier_merge:
            model.merge_classifier_cloud()
        model_log_file = args.output_dir + '/output.log'
        self.logger = setup_logger('default_logger', model_log_file, level=logging.DEBUG)
        self.device = device
        
        self.model = model # model是整个模型，会在sfl_simulator类中将该模型划分为client-side model和server-side model。
        self.criterion = criterion
        self.num_client = args.num_client
        self.num_epoch = args.num_epoch
        self.num_class = args.num_class
        self.batch_size = args.batch_size
        self.output_dir = args.output_dir
        self.div_lambda = [args.div_lambda for _ in range(args.num_client)]
        # 使用一个列表div_lambda，存放所有client的\lambda（用于divergence的计算）。初值为1.0。
        self.auto_scaler = True # 由于Divergence-aware方法中，\lambda_k仅在初次计算divergence时更新，因此用参数auto_scaler判断是否是初次计算divergence，之后置为False。
        self.client_sample_ratio  = args.client_sample_ratio

        # set dummy variables
        self.s_instance = None # server-side model
        self.c_instance_list = []  # 存放所有clients的model
        self.s_optimizer = None
        self.c_optimizer_list = []
        self.s_scheduler = None
        self.c_scheduler_list = []

        #initialize data iterator
        self.client_dataloader = train_loader
        self.validate_loader = test_loader
        self.client_iterator_list = []
        if train_loader is not None:
            for client_id in range(args.num_client):
                # train_loader[client_id].persistent_workers = True #TODO: remove if hurts
                self.client_iterator_list.append(create_iterator(iter((train_loader[client_id]))))
    
    # MocoSFL
    def next_data_batch(self, client_id, epoch = 0, is_distributed=False): # 获取client_id的train_loader中下一个batch的数据
        # 这里为什么要用iter()这么复杂的方式，而不直接用for循环？因为模型需要所有client同时遍历自己的数据，从而产生representation用于cut_layer的聚合，用for循环太复杂了。并且，每个client的batch都需要循环遍历，因此一定要使用iter()才能实现。
        try: 
            images, labels = next(self.client_iterator_list[client_id])
            if images.size(0) != self.batch_size:
                try: # 一定要写try-except，因为next的指针移到最后一个元素的下一位时会报错
                    next(self.client_iterator_list[client_id])
                except StopIteration:
                    pass
                if is_distributed:
                    self.client_dataloader[client_id].sampler.set_epoch(epoch)
                self.client_iterator_list[client_id] = create_iterator(iter((self.client_dataloader[client_id])))
                # create_iterator函数好像很多余，直接
                # self.client_iterator_list[client_id] = iter((self.client_dataloader[client_id]))感觉也可以？
                # self.client_dataloader== train_loader ==  get_cifar10_pairloader
                images, labels = next(self.client_iterator_list[client_id])
        except StopIteration: # 如果已经取完了，就从头开始取。
            if is_distributed:
                    self.client_dataloader[client_id].sampler.set_epoch(epoch)
            self.client_iterator_list[client_id] = create_iterator(iter((self.client_dataloader[client_id])))
            images, labels = next(self.client_iterator_list[client_id])
        return images, labels #由于遍历的是pairloader，所以images和labels为正例对两个augmented images。
    
    # FedSwav
    def next_swavdata_batch(self, client_id, use_dali, epoch = 0, local_rank=0, world_size=1, is_distributed = False):
        try: 
            images = next(self.client_iterator_list[client_id]) # images是个list，其中包含2+6个crops
#             pdb.set_trace()
            # 如果是dali读取,则images为包含'label'和'data'的dict
            if use_dali:
                bs = images['data'].size(0)
            else:
                bs = images[0].size(0)
            if bs != self.batch_size:
                try: # 一定要写try-except，因为next的指针移到最后一个元素的下一位时会报错
                    next(self.client_iterator_list[client_id])
                except StopIteration:
                    pass
                if use_dali:
                    self.client_dataloader[client_id].reset(epoch+1, num_shards=world_size, shard_id=local_rank, shuffle = True)
                    self.client_dataloader[client_id]._ever_consumed = False
                elif is_distributed:
                    self.client_dataloader[client_id].sampler.set_epoch(epoch)
                self.client_iterator_list[client_id] = create_iterator(iter((self.client_dataloader[client_id])))
                # create_iterator函数好像很多余，直接
                # self.client_iterator_list[client_id] = iter((self.client_dataloader[client_id]))感觉也可以？
                # self.client_dataloader== train_loader ==  get_cifar10_pairloader
                images = next(self.client_iterator_list[client_id])
        except StopIteration: # 如果已经取完了，就从头开始取。
            if use_dali:
                self.client_dataloader[client_id].reset(epoch+1, num_shards=world_size, shard_id=local_rank, shuffle = True)
                self.client_dataloader[client_id]._ever_consumed = False
            elif is_distributed:
                self.client_dataloader[client_id].sampler.set_epoch(epoch)
            self.client_iterator_list[client_id] = create_iterator(iter(self.client_dataloader[client_id]))
            images = next(self.client_iterator_list[client_id])
#         print(f"读取下一批的数据长度为{len(images.size())}")
        return images 

    # 对所有model取.zero_grad()
    def optimizer_zero_grads(self):  # This needs to be called
        if self.s_optimizer is not None:
            self.s_optimizer.zero_grad()
        if self.c_optimizer_list: 
            for i in range(self.num_client):
                self.c_optimizer_list[i].zero_grad()


    def fedavg(self, pool = None, divergence_aware = False, divergence_measure = False):
        # 参数divergence_measure的作用：判断是否计算divergence
        # 参数divergence_aware的作用：判断是否计算divergence，并使用由divergence计算得到的\mu对本轮client-side model进行动量更新。
        sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
        
        global_weights = average_weights(self.model.local_list, pool) 
        # 计算client-side model参数的均值
        if divergence_measure:
            divergence_list = []
        for i in range(self.num_client):
            
            if divergence_measure:
                if pool is None:
                    pool = range(len(self.num_client))
                
                if i in pool: # if current client is selected.
                    weight_divergence = 0.0
                    for key in global_weights.keys():
                        if "running" in key or "num_batches" in key: # skipping batchnorm running stats
                            continue
                            # 跳过BN层的running_mean,running_var, num_batches_tracked三个参数(why?)
                        weight_divergence += torch.linalg.norm(
                            torch.flatten(
                                self.model.local_list[i].state_dict()[key] - global_weights[key]).float(),
                                                               dim = -1, ord = 2)
                        # ord=2计算向量的最大奇异值，也就是向量的2范数。
                        # 第i个client的所有层的divergence求和，存入weight_divergence。
#                         dist, P, C = sinkhorn(self.model.local_list[i].state_dict()[key], global_weights[key])
#                         weight_divergence += dist.sum()
                    divergence_list.append(weight_divergence.item())

            if divergence_aware:
                '''
                [1]DAPU: Zhuang et al. - Collaborative Unsupervised Visual Representation Learning from Decentralized Data
                [2]Divergence_aware Zhuang et al. - Divergence-aware Federated Self-Supervised Learning

                Difference: [1] it is only used for the MLP predictor part in the online encoder (the fast one) in BYOL, not in any of the backbone model.
                            [2] it is used for the entire online encoder as well as its predictor. auto_scaler is invented.
                            
                '''
                if pool is None:
                    pool = range(len(self.num_client))
                
                if i in pool: # if current client is selected.
                    # 对于本轮训练选中的client，使用聚合参数进行动量更新
                    weight_divergence = 0.0
                    for key in global_weights.keys():
                        if "running" in key or "num_batches" in key: # skipping batchnorm running stats
                            continue
                        weight_divergence += torch.linalg.norm(torch.flatten(self.model.local_list[i].state_dict()[key] - global_weights[key]).float(), dim = -1, ord = 2)
#                         dist, P, C = sinkhorn(self.model.local_list[i].state_dict()[key], global_weights[key])
#                         weight_divergence += dist.sum()
                    mu = self.div_lambda[i] * weight_divergence.item() # the choice of dic_lambda depends on num_param in client-side model （所有div_lambda的初值为1.0)
                    mu = 1 if mu >= 1 else mu # If divergence is too large, just do personalization & don't consider the average.

                    for key in global_weights.keys(): # 使用聚合参数对local model的参数进行动量更新
                        self.model.local_list[i].state_dict()[key] = mu * self.model.local_list[i].state_dict()[key] + (1 - mu) * global_weights[key]

                    if self.auto_scaler: # is only done at epoch 1
                        # 仅初次计算divergence时计算\lambda_k（初次计算应该把所有client都包含在内）
                        self.div_lambda[i] = mu / weight_divergence # such that next div_lambda will be similar to 1. will not be a crazy value.（原论文中的\tao设定为\mu）
                        self.auto_scaler = False # Will only use it once at the first round.
                else: # if current client is not selected.（i not in pool）
                    # 对于本轮没选中的client，直接使用聚合参数进行更新。（不使用动量更新）
                    self.model.local_list[i].load_state_dict(global_weights)
            else: # divergence_aware=False:不使用divergence-aware的动量更新,直接使用聚合参数更新。
                '''Normal case: directly get the averaged result'''
#                 mu=0.1
#                 for key in global_weights.keys(): # 使用聚合参数对local model的参数进行动量更新
#                     self.model.local_list[i].state_dict()[key] = mu * self.model.local_list[i].state_dict()[key] + (1 - mu) * global_weights[key]
                self.model.local_list[i].load_state_dict(global_weights)

        if divergence_measure: # 如果计算了divergence,则函数返回所有client-side model的divergence
            return divergence_list
        else:
            return None
        
    def train(self): # 将所有model的模式设置为train mode。
        if self.c_instance_list: 
            for i in range(self.num_client):
                self.c_instance_list[i].train() # 即model.train()，
        if self.s_instance is not None:
            self.s_instance.train()

    def eval(self): # 将所有model的模式设置为eval mode。
        if self.c_instance_list: 
            for i in range(self.num_client):
                self.c_instance_list[i].eval()
        if self.s_instance is not None:
            self.s_instance.eval()
    
    def cuda(self, device): # 将所有model的加载到cuda
        if self.c_instance_list: 
            for i in range(self.num_client):
                self.c_instance_list[i].cuda(device)
        if self.s_instance is not None:
            self.s_instance.cuda(device)

    def cpu(self): # 将所有model的加载到cpu
        if self.c_instance_list: 
            for i in range(self.num_client):
                self.c_instance_list[i].cpu()
        if self.s_instance is not None:
            self.s_instance.cpu()

    def validate(self): # validate in cuda mode
        """
        Run evaluation（用于训练阶段的测试，返回模型在验证集上的准确率）
        """
        top1 = AverageMeter()
        self.eval()  # set to eval mode
        if self.c_instance_list:
            self.c_instance_list[0].cuda(self.device)
        if self.s_instance is not None:
            self.s_instance.to(self.device)
        
        if self.c_instance_list:
            for input, target in self.validate_loader:
                input = input.to(self.device)
                target = target.to(self.device)
                with torch.no_grad():
                    output = self.c_instance_list[0](input) # local model计算得到表征
                    if self.s_instance is not None:
                        output = self.s_instance(output) # server model在表征的基础上做分类
                prec1 = accuracy(output.data, target)[0] 
                # 训练的时候的验证只计算top1准确率（最终测试才可以降低标准到topk）
                top1.update(prec1.item(), input.size(0)) 
                # 计算所有验证集中所有batch正确率的加权平均，权重为batch_size

        self.train() #set back to train mode
        return top1.avg # 返回正确率的均值

    def save_model(self, epoch, is_best=False, is_distributed=False): # 保存模型
        if is_best:
            epoch = "best"
        if is_distributed:
            torch.save(self.model.cloud.module.state_dict(), self.output_dir + f'/checkpoint_s_{epoch}.tar')
            torch.save(self.model.local_list[0].module.state_dict(), self.output_dir + f'/checkpoint_c_{epoch}.tar')
        else:
            torch.save(self.model.cloud.state_dict(), self.output_dir + f'/checkpoint_s_{epoch}.tar')
            torch.save(self.model.local_list[0].state_dict(), self.output_dir + f'/checkpoint_c_{epoch}.tar')
        # 只保存第0个client-side model
        
    
    def load_model(self, is_best=True, epoch=200): # 加载最优模型
        if is_best:
            epoch = "best"
        checkpoint_s = torch.load(self.output_dir + f'/checkpoint_s_{epoch}.tar')
        self.model.cloud.load_state_dict(checkpoint_s)
        checkpoint_c = torch.load(self.output_dir + f'/checkpoint_c_{epoch}.tar')
        for i in range(self.num_client):
            self.model.local_list[i].load_state_dict(checkpoint_c)
            # 使用保存的第0个client-side model加载所有client-side models

    def load_model_from_path(self, model_path, load_client = True, load_server = False):
        # 加载最优模型
        if load_server:
            checkpoint_s = torch.load(model_path + '/checkpoint_s_best.tar')
            self.model.cloud.load_state_dict(checkpoint_s)
        if load_client:
            checkpoint_c = torch.load(model_path + '/checkpoint_c_best.tar')
            for i in range(self.num_client):
                self.model.local_list[i].load_state_dict(checkpoint_c)

    def log(self, message):
        self.logger.debug(message)

class create_iterator(): # 创建一个迭代器实体，该实体包括next方法。
    def __init__(self, iterator) -> None:
        self.iterator = iterator

    def __next__(self):
        return next(self.iterator)

class create_base_instance: # 创建一个model实体, 该实体包含设置两种mode, 加载到cuda和cpu的方法
    def __init__(self, model) -> None:
        self.model = model

    def __call__(self):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def cuda(self, device):
        self.model.to(device)
    
    def cpu(self):
        self.model.cpu()