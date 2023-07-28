'''
SFL-mocoV1 basic functionality, wrap in a class. 

Can be extended to real-applcations if communication protocol is considered.

To understand how the training works in this implementation of SFL. We provide a tutorial in __main__ function:

Refer to He et al. Momentum Contrast for Unsupervised Visual Representation Learning for technical details.

'''

import torch
import copy
import math
import torch.nn as nn
from functions.base_funtions import base_simulator, create_base_instance
import torchvision.transforms as transforms
import torch.nn.functional as F
from models.resnet import init_weights
from utils import AverageMeter, accuracy
import numpy as np
import pdb


class sflmoco_simulator(base_simulator):
    def __init__(self, model, criterion, train_loader, test_loader, args) -> None:
        super().__init__(model, criterion, train_loader, test_loader, args)
        
        # Create server instances
        if self.model.cloud is not None:
            # self.s_instance是server-side model
            print(self.model.prototypes)
            self.prototypes = self.model.get_prototypes()
#             print(self.prototypes)
            self.s_instance = create_sflmocoserver_instance(self.model.cloud, 
                                                            self.prototypes,
                                                            criterion, 
                                                            args, 
                                                            self.model.get_smashed_data_size(1, args.data_size), 
                                                            feature_sharing=args.feature_sharing)
            self.s_optimizer = torch.optim.SGD(list(self.s_instance.model.parameters()), 
                                               lr=args.lr, 
                                               momentum=args.momentum, 
                                               weight_decay=args.weight_decay)
            
            if args.cos:
                self.s_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.s_optimizer, self.num_epoch)  # learning rate decay 
            else:
                milestones = [int(0.6*self.num_epoch), int(0.8*self.num_epoch)]
                self.s_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.s_optimizer, milestones=milestones, gamma=0.1)  # learning rate decay 

        # Create client instances
        self.c_instance_list = [] # self.c_instance_list是所有client-side model的list
        for i in range(args.num_client):
            self.c_instance_list.append(create_sflmococlient_instance(self.model.local_list[i]))

        self.c_optimizer_list = [None for i in range(args.num_client)]
        for i in range(args.num_client):
            self.c_optimizer_list[i] = torch.optim.SGD(list(self.c_instance_list[i].model.parameters()), lr=args.c_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        self.c_scheduler_list = [None for i in range(args.num_client)]
        if args.cos:
            for i in range(args.num_client):
                self.c_scheduler_list[i] = torch.optim.lr_scheduler.CosineAnnealingLR(self.c_optimizer_list[i], self.num_epoch)  # learning rate decay
        else:
            milestones = [int(0.6*self.num_epoch), int(0.8*self.num_epoch)]
            for i in range(args.num_client):
                self.c_scheduler_list[i] = torch.optim.lr_scheduler.MultiStepLR(self.c_optimizer_list[i], milestones=milestones, gamma=0.2)  # learning rate decay
        # Set augmentation
        self.K_dim = args.K_dim
        self.data_size = args.data_size
        self.arch = args.arch
        
    def linear_eval(self, memloader, num_epochs = 100, lr = 3.0, use_dali=False): # Use linear evaluation
        """
        Run Linear evaluation（linear_eval用于最终的测试）
        """
        self.cuda(self.device) # self是个simulator，base_simulator中对simulator.cuda()或.eval()等的定义：遍历self的所有model，并执行.cuda()或.eval()操作。
        self.eval()  #set to eval mode
        criterion = nn.CrossEntropyLoss()

        self.model.unmerge_classifier_cloud() # 将原模型的分类层去掉，训练一个新的分类层。

        # if self.data_size == 32:
        #     data_size_factor = 1
        # elif self.data_size == 64:
        #     data_size_factor = 4
        # elif self.data_size == 96:
        #     data_size_factor = 9
        # classifier_list = [nn.Linear(self.K_dim * self.model.expansion, self.num_class)]

        if "ResNet" in self.arch or "resnet" in self.arch:
            if "resnet" in self.arch:
                self.arch = "ResNet" + self.arch.split("resnet")[-1]
            output_dim = 512
        elif "vgg" in self.arch:
            output_dim = 512
        elif "MobileNetV2" in self.arch:
            output_dim = 1280

        classifier_list = [nn.Linear(output_dim * self.model.expansion, self.num_class)]
        linear_classifier = nn.Sequential(*classifier_list)

        linear_classifier.apply(init_weights)

        # linear_optimizer = torch.optim.SGD(list(linear_classifier.parameters()), lr=lr, momentum=0.9, weight_decay=1e-4)
        linear_optimizer = torch.optim.Adam(list(linear_classifier.parameters()))
        linear_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(linear_optimizer, num_epochs//4)  # learning rate decay 

        linear_classifier.cuda(self.device)
        linear_classifier.train()
        
        best_avg_accu = 0.0
        avg_pool = nn.AdaptiveAvgPool2d((1,1))
        # Train the linear layer
        for epoch in range(num_epochs):
            if use_dali is True:
                for input in memloader[0]:
                    data = input['data'].to(self.device)
                    label = input['label'].t().squeeze().to(self.device)
                    linear_optimizer.zero_grad()
                    with torch.no_grad():
                        output = self.model.local_list[0](data)
                        output = self.model.cloud(output)
                        output = avg_pool(output) #[B, output_dim * self.model.expansion, 1]
                        output = output.view(output.size(0), -1)  # [B,output_dim * self.model.expansion]
                    output = linear_classifier(output.detach()) # [B,num_classes]
                    # 这里output要detach，因为只训练线性分类层，反向传播只计算分类层的梯度。
                    loss = criterion(output, label)
                    # loss = loss_xent(output, label)
                    loss.backward()
                    linear_optimizer.step()
                    linear_scheduler.step()
            else:
                for input, label in memloader[0]:
                    input = input.to(self.device)
                    label = label.to(self.device)
                    linear_optimizer.zero_grad()
                    with torch.no_grad():
                        output = self.model.local_list[0](input)
                        output = self.model.cloud(output)
                        output = avg_pool(output) #[B, output_dim * self.model.expansion, 1]
                        output = output.view(output.size(0), -1)  # [B,output_dim * self.model.expansion]
                    output = linear_classifier(output.detach()) # [B,num_classes]
                    # 这里output要detach，因为只训练线性分类层，反向传播只计算分类层的梯度。
                    loss = criterion(output, label)
                    # loss = loss_xent(output, label)
                    loss.backward()
                    linear_optimizer.step()
                    linear_scheduler.step()
            
            """
            Run validation
            """
            top1 = AverageMeter()
            
            linear_classifier.eval()
            if use_dali is True:
                for input in self.validate_loader[0]:
                    data = input['data'].to(self.device)
                    target = input['label'].t().squeeze().to(self.device)
                    with torch.no_grad():
                        output = self.model.local_list[0](data)
                        output = self.model.cloud(output)
                        output = avg_pool(output)
                        output = output.view(output.size(0), -1)
                        output = linear_classifier(output.detach())
                    prec1 = accuracy(output.data, target)[0]
                    top1.update(prec1.item(), data.size(0)) # 计算目前遍历的所有batch的均值
            else:
                for input, target in self.validate_loader:
                    input = input.to(self.device)
                    target = target.to(self.device)
                    with torch.no_grad():
                        output = self.model.local_list[0](input)
                        output = self.model.cloud(output)
                        output = avg_pool(output)
                        output = output.view(output.size(0), -1)
                        output = linear_classifier(output.detach())
                    prec1 = accuracy(output.data, target)[0]
                    top1.update(prec1.item(), input.size(0)) # 计算目前遍历的所有batch的均值
            linear_classifier.train()
            avg_accu = top1.avg
            if avg_accu > best_avg_accu:
                best_avg_accu = avg_accu
            print(f"Epoch: {epoch}, linear eval accuracy - current: {avg_accu:.2f}, best: {best_avg_accu:.2f}")
        
        self.model.merge_classifier_cloud() 
        # 由于训练好的linear_classifier并没有赋值给self.classifier，因此这一步仍然将训练过程中的分类层加到模型上。
        self.train()  #set back to train mode
        return best_avg_accu


    def semisupervise_eval(self, memloader, num_epochs = 100, lr = 3.0, use_dali=False): # Use semi-supervised learning as evaluation
        """
        Run Linear evaluation
        """
        self.cuda(self.device)
        self.eval()  #set to eval mode
        criterion = nn.CrossEntropyLoss().to(self.device)

        self.model.unmerge_classifier_cloud()

        classifier_list = [nn.Linear(512 * self.model.expansion, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(True),
                            nn.Linear(512, self.num_class)]
        semi_classifier = nn.Sequential(*classifier_list)

        semi_classifier.apply(init_weights)

        # linear_optimizer = torch.optim.SGD(list(semi_classifier.parameters()), lr=lr, momentum=0.9, weight_decay=1e-4)
        linear_optimizer = torch.optim.Adam(list(semi_classifier.parameters()), lr=1e-3) # as in divergence-aware
        milestones = [int(0.6*num_epochs), int(0.8*num_epochs)]
        linear_scheduler = torch.optim.lr_scheduler.MultiStepLR(linear_optimizer, milestones=milestones, gamma=0.1)  # learning rate decay 

        semi_classifier.cuda(self.device)
        semi_classifier.train()
        avg_pool = nn.AdaptiveAvgPool2d((1,1))
        best_avg_accu = 0.0
        # Train the linear layer
        for epoch in range(num_epochs):
            if use_dali is True:
                for input in memloader[0]:
                    data = input['data'].to(self.device)
                    label = input['label'].t().squeeze().to(self.device)
                    with torch.no_grad():
                        output = self.model.local_list[0](data)
                        output = self.model.cloud(output)
                        # output = F.avg_pool2d(output, 4)
                        output = avg_pool(output)
                        output = output.view(output.size(0), -1)
                    output = semi_classifier(output.detach())
                    loss = criterion(output, label)
                    # loss = loss_xent(output, label)
                    loss.backward()
                    linear_optimizer.step()
                    linear_scheduler.step()
            else:
                for input, label in memloader[0]:
                    linear_optimizer.zero_grad()
                    input = input.to(self.device)
                    label = label.to(self.device)
                    with torch.no_grad():
                        output = self.model.local_list[0](input)
                        output = self.model.cloud(output)
                        # output = F.avg_pool2d(output, 4)
                        output = avg_pool(output)
                        output = output.view(output.size(0), -1)
                    output = semi_classifier(output.detach())
                    loss = criterion(output, label)
                    # loss = loss_xent(output, label)
                    loss.backward()
                    linear_optimizer.step()
                    linear_scheduler.step()
            
            """
            Run validation
            """
            top1 = AverageMeter()
            
            semi_classifier.eval()
            if use_dali is True:
                for input in self.validate_loader[0]:
                    data = input['data'].to(self.device)
                    target = input['label'].t().squeeze().to(self.device)
                    with torch.no_grad():
                        output = self.model.local_list[0](data)
                        output = self.model.cloud(output)
                        # output = F.avg_pool2d(output, 4)
                        output = avg_pool(output)
                        output = output.view(output.size(0), -1)
                        output = semi_classifier(output.detach())

                    prec1 = accuracy(output.data, target)[0]
                    top1.update(prec1.item(), data.size(0))
            else:
                for input, target in self.validate_loader:
                    input = input.to(self.device)
                    target = target.to(self.device)
                    with torch.no_grad():
                        output = self.model.local_list[0](input)
                        output = self.model.cloud(output)
                        # output = F.avg_pool2d(output, 4)
                        output = avg_pool(output)
                        output = output.view(output.size(0), -1)
                        output = semi_classifier(output.detach())

                    prec1 = accuracy(output.data, target)[0]
                    top1.update(prec1.item(), input.size(0))
            semi_classifier.train()
            avg_accu = top1.avg
            if avg_accu > best_avg_accu:
                best_avg_accu = avg_accu
            print(f"Epoch: {epoch}, linear eval accuracy - current: {avg_accu:.2f}, best: {best_avg_accu:.2f}")
        
        self.model.merge_classifier_cloud()
        self.train()  #set back to train mode
        return best_avg_accu

    def knn_eval(self, memloader, use_dali=False): # Use linear evaluation memloader作为KNN的已知样本点数据集
        '''用于每个epoch训练之后的验证'''
        if self.c_instance_list:
            self.c_instance_list[0].cuda(self.device)
        # test using a knn monitor
        
        def test():
            self.eval() # 将self的所有model设置为eval mode
            classes = self.num_class
            total_top1, total_top5, total_num, feature_bank, feature_labels = 0.0, 0.0, 0, [], []
            with torch.no_grad():
                # generate feature bank
                if use_dali is True:
                    for input in memloader[0]: # memloader是一个list，其中的元素才是DataLoader。
                        data = input['data'].to(self.device, non_blocking=True)
                        target = input['label'].t().squeeze() #dali读取的labels是二维的, pytorch的dataloader中的label是一维的.
#                         print(next(self.model.parameters()).device)
                        feature = self.model(data) # self.model的forward函数默认使用client_id=0，即client-side model使用第0个模型。(resnet.py)
                        feature = F.normalize(feature, dim=1)
                        feature_bank.append(feature)
                        feature_labels.append(target)  
                else:
                    for data, target in memloader[0]: # memloader是一个list，其中的元素才是DataLoader。
                        feature = self.model(data.to(self.device, non_blocking=True)) # self.model的forward函数默认使用client_id=0，即client-side model使用第0个模型。(resnet.py)
                        feature = F.normalize(feature, dim=1)
                        feature_bank.append(feature)
                        feature_labels.append(target) 
                    # target和feature都是tensor([x,x,x])的形式，因此此循环结束时的feature_labels为[tensor([x,x,,...],tensor([x,x,...],...))]的形式。而之后的torch.cat(feature_labels,dim=0)操作会将feature_labels列表中的所有tensor合并为一个tensor，即tensor([x,x,x,...])
                # [D, N] (做了个转置，方便后面做乘法。每列代表一条数据的表征)
                feature_bank = torch.cat(feature_bank, dim=0).t().contiguous().to(self.device)
                # [N]
                feature_labels = torch.cat(feature_labels, dim=0).contiguous().to(self.device)
                # feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
                # loop test data to predict the label by weighted knn search
                
                if use_dali is True:
                    for input in self.validate_loader[0]: 
                        data = input['data'].to(self.device, non_blocking=True)
                        target = input['label'].t().squeeze().to(self.device, non_blocking=True)
                        feature = self.model(data)
                        feature = F.normalize(feature, dim=1)
                        pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, 200, 0.1)
                        # 参数200表示找出距离最近的200个已知样本点
                        # pred_labels为预测的类别结果，其中第0列表示出现次数最多的类别。

                        total_num += data.size(0) # total_num计算validate_loader中所有数据的条数
                        total_top1 += (pred_labels[:, 0] == target).float().sum().item() 
                        
                else:
                    for data, target in self.validate_loader:
                        data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                        # 这里non_blocking=True使得data和target的加载可以并行进行
                        feature = self.model(data)
                        feature = F.normalize(feature, dim=1)

                        pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, 200, 0.1)
                        # 参数200表示找出距离最近的200个已知样本点
                        # pred_labels为预测的类别结果，其中第0列表示出现次数最多的类别。

                        total_num += data.size(0) # total_num计算validate_loader中所有数据的条数
                        total_top1 += (pred_labels[:, 0] == target).float().sum().item() 
                        # total_top1计算整个validate_loader中top1_acc正确的个数
                        # print('KNN Test: Acc@1:{:.2f}%'.format(total_top1 / total_num * 100))

            return total_top1 / total_num * 100 # 返回top1准确率

        # knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
        # implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
        def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # 每条数据都与feature_bank中的表征计算内积，得到该数据与所有N个已知样本点的距离。
            # 对每条数据的距离结果进行排序并取值最大的k个，sim_indices中保存排序后元素对应的下标
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
            # 找到距离最近的200个样本点的类别
            sim_weight = (sim_weight / knn_t).exp() # 距离扩大10倍然后取指数，将距离差距扩大

            # counts for each class
            one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
            # classes表示类别的数量。每个feature对应knn_k个已知样本点，每个样本点都对应classes个类别中的一个。由此构造一个one-hot矩阵。
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
            # .unsqueeze(dim=-1)是在最后一维后面增加一个维度，[B, K]-->[B, K, 1]。为什么要增加一维？因为one_hot_label.view(feature.size(0), -1, classes)的维度重置： [B*K, C]-->[B, K, C]。
            # 乘法的结果：将sim_weight中的距离值与one-hot矩阵中的对应位置相乘，从而将one-hot矩阵变成距离权重矩阵。（当存在出现次数一样多的类别时无法判断最终类别。可以通过距离进一步对出现次数进行放大，从而出现次数相同但距离更近的类别作为最终类别。）
            # .sum(...,dim=1)：对每个batch的one-hot矩阵进行行求和，即统计类别的加权出现次数。结果的维度为[B, C]，表示B(batch_size)个特征，距离每个特征最近的200个样本点的类别出现次数信息。

            pred_labels = pred_scores.argsort(dim=-1, descending=True) # 对出现次数信息的最后一维降序排列，并返回排序后的元素在原矩阵中的索引。这里的索引就是类别，从而第0列即top1(出现次数最多的)类别，第1列为出现次数第2多的类别, ...
            return pred_labels
        
        test_acc_1 = test()
        self.train() #set back to train
        return test_acc_1

class create_sflmocoserver_instance(create_base_instance):
    def __init__(self, model, prototypes, criterion, args, server_input_size = 1, feature_sharing = True) -> None:
        super().__init__(model)
        self.criterion = criterion
        self.device = args.device
        self.t_model = copy.deepcopy(model) 
        # 这是创建server_instance的类，传入的model是server-side model
        self.prototypes = prototypes
        self.symmetric = args.symmetric
        self.batch_size = args.batch_size
        self.num_client = args.num_client
        for param_t in self.t_model.parameters():
            param_t.requires_grad = False  # not update by gradient
            #为什么server-side model不用梯度更新？

        self.K = args.K #max number of keys stored in queue，即queue的最大容量
        self.T = args.T #Temperature of InfoCE loss
        
        self.use_the_queue = False #用于表示当前的分配过程是否使用队列

        self.feature_sharing = feature_sharing
        if self.feature_sharing:
#             self.queue = torch.randn(len(args.crops_for_assign), args.K_dim, self.K).to(self.device) #K_dim: key中向量的维度
            #队列的初值为随机值（这个队列是横着的，每个feature是竖着的）
#             self.queue = nn.functional.normalize(self.queue, dim=1) # 对队列中的每个列向量标准化
#             pdb.set_trace()
            self.queue = torch.zeros(len(args.crops_for_assign), args.K_dim, self.K).to(self.device) 
            self.queue_ptr = torch.zeros(2, dtype=torch.long) #queue_ptr表示队列的指针，初值为[0]
        else: # 如果不进行特征聚合共享，则每个client都维护一个自己的队列
            self.K = self.K // self.num_client #将总队列容量均匀分给每个client的队列
#             self.queue = []
#             self.queue_ptr = []
            self.queue = [[] for _ in len(args.crops_for_assign)]
            self.queue_ptr = [[] for _ in len(args.crops_for_assign)]
            for _ in range(self.num_client):
#                 self.queue.append(torch.randn(args.K_dim, self.K).to(self.device)) # queue中包含了若干个子队列
#                 self.queue_ptr.append(torch.zeros(1, dtype=torch.long))
                 for i in range(args.crops_for_assign):
                    self.queue[i].append(torch.zeros(args.K_dim, self.K).to(self.device)) 
                    self.queue_ptr[i].append(torch.zeros(1, dtype=torch.long))
                
    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        print("调用server_instance的forward函数")
        output = self.model(input)
#         if self.prototypes is not None:
#             return output, self.prototypes(output)
        return output

    
    @torch.no_grad()
    def _dequeue_and_enqueue_swav(self, keys, crop_index, pool = None):
        # 参数crop_index表示当前传入的是哪类(0,1)增强样本
        # gather keys before updating queue
        if self.feature_sharing: # feature_sharing这个参数默认为True
            batch_size = keys.shape[0]
            ptr = int(self.queue_ptr[crop_index])
            
            # replace the keys at ptr (dequeue and enqueue)（左边队首，右边队尾）
            if (ptr + batch_size) <= self.K: # 指针没超过queue的最大容量，则将keys的转置加在队尾
#                 self.queue[:, ptr:ptr + batch_size] = keys.T
                self.queue[crop_index, :, ptr:ptr + batch_size] = keys.T
            else: #队列已满
                self.queue[crop_index, :, ptr:] = keys.T[:, :self.K - ptr] 
                # 队列中还有self.K-ptr个空位置，则只放keys中的前self.K-ptr个features入队
                self.queue[crop_index, :, 0:(batch_size + ptr - self.K)] = keys.T[:, self.K - ptr:]
                # 按先进先出原则，将队首的self.K-ptr个位置用于存放keys中剩下的features(循环队列的结构)
            ptr = (ptr + batch_size) % self.K  # move pointer，指针指到当前队尾(即keys中最后一个feature所在的位置)

            self.queue_ptr[crop_index][0] = ptr # 保存指针位置(queue_ptr中只有一个元素)
            

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, pool = None):
        # gather keys before updating queue
        if self.feature_sharing: # feature_sharing这个参数默认为True
            batch_size = keys.shape[0]
            ptr = int(self.queue_ptr)
            
            # replace the keys at ptr (dequeue and enqueue)
            if (ptr + batch_size) <= self.K: # 指针没超过queue的最大容量，则将keys的转置加在队尾
                self.queue[:, ptr:ptr + batch_size] = keys.T
            else: #队列已满
                self.queue[:, ptr:] = keys.T[:, :self.K - ptr] 
                # 队列中还有self.K-ptr个空位置，则只放keys中的前self.K-ptr个features入队
                self.queue[:, 0:(batch_size + ptr - self.K)] = keys.T[:, self.K - ptr:]
                # 按先进先出原则，将队首的self.K-ptr个位置用于存放keys中剩下的features(循环队列的结构)
            ptr = (ptr + batch_size) % self.K  # move pointer，指针指到当前队尾(即keys中最后一个feature所在的位置)

            self.queue_ptr[0] = ptr # 保存指针位置(queue_ptr中只有一个元素)
        else:
            batch_size = self.batch_size
            if pool is None:
                pool = range(self.num_client)
            for client_id in pool: # 将keys均分给每个client
                client_key = keys[client_id*batch_size:(client_id + 1)*batch_size] 
                ptr = int(self.queue_ptr[client_id])
                # replace the keys at ptr (dequeue and enqueue)
                if (ptr + batch_size) <= self.K:
                    self.queue[client_id][:, ptr:ptr + batch_size] = client_key.T # client_id的key入队
                else: 
                    self.queue[client_id][:, ptr:] = client_key.T[:, :self.K - ptr]
                    self.queue[client_id][:, 0:(batch_size + ptr - self.K)] = client_key.T[:, self.K - ptr:]
                ptr = (ptr + batch_size) % self.K  # move pointer
                self.queue_ptr[client_id][0] = ptr # 保存client_id的队尾位置

    @torch.no_grad()
    def update_moving_average(self, tau = 0.99): 
        # 按照MoCo, 用于计算key的encoder采用动量更新。
        for online, target in zip(self.model.parameters(), self.t_model.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm. 
        为什么需要打乱batch的顺序？原本pkey是按client的顺序排列的，如果直接在dim=1维做BN，则是在每个client内部的所有表征向量上进行BN。将pkey的每条打乱
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).to(self.device)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle) # idx_unshuffle表示还原batch顺序所对应的下标

        return x[idx_shuffle], idx_unshuffle #返回打乱后的表征向量，以及这些向量在原batch中的下标。

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """Undo batch shuffle.  还原batch的顺序。"""
        return x[idx_unshuffle]
    
    @torch.no_grad()
    def distributed_sinkhorn(self, out, epsilon, sinkhorn_iterations): # 加了注解@torch.no_grad()的全都是类的内部方法，需要用self.调用
        Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        return Q.t()


    def contrastive_loss(self, query, pkey, pool = None): 
        # 在contrastive_loss函数中，会使用server端的网络对已有表征做进一步的表征，得到query_out和pkey_out，并在此基础上计算对比loss。最后，返回计算得到的对比loss和两个表征。
        #正例对为(query,pkey)，负例对为(query,queue)
        query_out = self.model(query) 
        # 这里的model是server-side model。在已有的query表征的基础上计算进一步表征

        query_out = nn.functional.normalize(query_out, dim = 1) 
        # query标准化后再用于计算余弦相似度

        with torch.no_grad():  # no gradient to keys

            pkey_, idx_unshuffle = self._batch_shuffle_single_gpu(pkey) 
            #pkey_表示打乱后的keys。打乱pkey是为了计算BN。

            pkey_out = self.t_model(pkey_) # 根据已有pkey表征计算keys的进一步表征

            pkey_out = nn.functional.normalize(pkey_out, dim = 1).detach() 
            # 对keys标准化，即BN。第1维是特征维，即对每个client中的所有表征在每个特征维对应归一化。（打乱好像没有起到任何作用？）

            pkey_out = self._batch_unshuffle_single_gpu(pkey_out, idx_unshuffle) # 还原keys的顺序

        l_pos = torch.einsum('nc,nc->n', [query_out, pkey_out]).unsqueeze(-1) 
        # 正例对中的query和key均进行归一化，以消除长度对结果的影响，之后计算内积（行对应乘）。此时的内积完全表示两向量的相关性。（否则，如果没有归一化，可能越长的向量值越大）
        
        if self.feature_sharing:
            l_neg = torch.einsum('nc,ck->nk', [query_out, self.queue.clone().detach()])#计算负例对内积
        else:
            if pool is None:
                pool = range(self.num_client)
            l_neg_list = []
            for client_id in pool: #既然feature_sharing==False, 为什么query_out能被client共享？
                l_neg_list.append(torch.einsum('nc,ck->nk', [query_out[client_id*self.batch_size:(client_id + 1)*self.batch_size], self.queue[client_id].clone().detach()]))
                # 把
            l_neg = torch.cat(l_neg_list, dim = 0)

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        loss = self.criterion(logits, labels) #CELoss

        accu = accuracy(logits, labels) 
        #这里的acc不是图片预测的acc，而是计算正例对的logits比负例对logits大的acc，记为contrast acc.

        return loss, accu, query_out, pkey_out
    
    
#     def compute(self, query, pkey, update_momentum = True, enqueue = True, tau = 0.99, pool = None):
#         # compute函数用于server端网络在已有表征的基础上计算对比loss，并针对server端的网络反向传播计算server端网络的梯度。同时，将keys的最终表征pkey_out入队作为负例。
#         query.requires_grad=True # 为什么需要修改.requires_grad参数？因为传入的query是经过.detach()的，requires_grad=False，需要修改为True，并以query为叶子节点构建新的计算图。

#         query.retain_grad() #.retain_grad()的目的是保留query的梯度信息。为什么要保留？在梯度计算图中反向传播计算完梯度之后，只有叶子节点的梯度信息会被保留，而非叶子节点的梯度信息会被清空，以节省空间。在SL中query的梯度信息需要被用于client端网络的反向传播，而query虽然在逻辑上是叶子节点，但实际上是一个中间节点，因此需要额外设置以保留。

#         if update_momentum:
#             self.update_moving_average(tau)

#         if self.symmetric: # 使用互信息计算
#             loss12, accu, q1, k2 = self.contrastive_loss(query, pkey, pool)
#             loss21, accu, q2, k1 = self.contrastive_loss(pkey, query, pool)
#             loss = loss12 + loss21
#             pkey_out = torch.cat([k1, k2], dim = 0) # 将同一张图片的两个aug的keys都入队作为负例
#         else:
#             loss, accu, query_out, pkey_out = self.contrastive_loss(query, pkey, pool)

#         if enqueue:
#             self._dequeue_and_enqueue(pkey_out, pool) # 使用当前batch的keys更新队列

#         error = loss.detach().cpu().numpy() # error变量用于记录loss的数值

#         if query.grad is not None:
#             query.grad.zero_() # 由于query涉及到与client端网络和server端网络相关的两个计算图，因此在server端先将query的梯度清零，
        
#         # loss.backward(retain_graph = True)
#         loss.backward()

#         gradient = query.grad.detach().clone() # get gradient, the -1 is important, since updates are added to the weights in cpp.
#         # 为什么要保存query的梯度信息？因为需要传回client端网络，使各client端网络使用query的梯度信息对自己的网络进行更新。

#         return error, gradient, accu[0] # accu是一个list，accu[0]表示top1 accuracy。

#     def compute_swav1(self, client_embedding_list, crops_for_assign, nmb_crops, temperature, epsilon, sinkhorn_iterations, enqueue = True, pool = None):
#         # compute函数用于server端网络在已有表征的基础上计算对比loss，并针对server端的网络反向传播计算server端网络的梯度。同时，将keys的最终表征pkey_out入队作为负例。
#         # normalize the prototypes
# #         print(f"server-side prototypes：{self.model}")
#         with torch.no_grad():
#             w = self.prototypes.weight.data.clone()
#             w = nn.functional.normalize(w, dim=1, p=2)
#             self.prototypes.weight.copy_(w)
#         self.prototypes.to(self.device)
            
#         gradient = [None for _ in range(len(pool))]
#         query_0 = [] # 保存所有client的标准增强样本的tensor
#         query_1 = [] # 保存所有client的其余增强样本的tensor
#         bs = []
#         # ============ swav loss ... ============
#         # 先将所有client的embedding合并为一个tensor
#         for client_index, client_id in enumerate(pool):
#             query_i_list = client_embedding_list[client_index] # query_i_list是两个tensor组成的list
#             bs.append(int(query_i_list[0].size(0)/nmb_crops[0]))
#             for crop_id in range(len(nmb_crops)): #crop_id==0,1
#                 query_crop_id = query_i_list[crop_id].detach() 
#                 # query_crop_id是一个tensor，包含2/6个相同size的增强样本的表征
#                 if crop_id==0: #标准增强样本
#                     query_0.append(query_crop_id)
#                 elif len(nmb_crops)==2:
#                     query_1.append(query_crop_id)
#                 else: # multi-crop做了3种size及以上的增强
#                     pass
                    
#         query_0 = torch.cat(query_0, dim=0) #size==[B*num_crops[0]*len(pool), args.K_dim]
#         query_0.requires_grad=True
#         query_0.retain_grad()
#         query_1 = torch.cat(query_1, dim=0)#size==[B*num_crops[1]*len(pool), args.K_dim]
        
#         # 计算两类增强样本的进一步表征
# #         pdb.set_trace()
# #         embedding_0 = self.model(query_0.to(self.device, non_blocking=True))
# #         embedding_1 = self.model(query_1.to(self.device, non_blocking=True))
        
# #         # 用prototypes层处理，计算两类增强样本表征的分类scores。
# #         score_0 = self.prototypes(embedding_0) #用prototypes层处理，得到所有表征的分类scores。
# #         score_1 = self.prototypes(embedding_1) #用prototypes层处理，得到所有表征的分类scores。

#         total_loss = 0 # 所有clients的loss之和
#         loss_i = 0 # 第i个client的loss
#         for client_index, client_id in enumerate(pool):
#             bs_i = bs[client_index]
#             pdb.set_trace()
#             query_0.to(self.device) # query_0有3800M
#             query_i_0 = query_0[bs_i * (client_index*nmb_crops[0]): bs_i * ((client_index+1)*nmb_crops[0])].to(self.device, non_blocking=True)
#             query_i_1 = query_1[bs_i * (client_index*nmb_crops[1]): bs_i * (client_index + 1)*nmb_crops[1]].to(self.device, non_blocking=True)
#             embedding_0 = self.model(query_i_0) # embedding_0有3800M
#             embedding_1 = self.model(query_i_1) # embedding_1有3800M
#             pdb.set_trace()

#             # 用prototypes层处理，计算两类增强样本表征的分类scores。
#             score_0 = self.prototypes(embedding_0) # 第一类标准增强样本的scores。embedding_1有3800M
#             score_1 = self.prototypes(embedding_1) # 第二类非标准增强样本的scores
            
#             for queue_i, crop_id in enumerate(crops_for_assign): #crop_id==0,1
#                 with torch.no_grad():
#                     embedding = embedding_0[bs_i * crop_id: bs_i * (crop_id + 1)]
#                     score = score_0[bs_i * crop_id: bs_i * (crop_id + 1)].detach()
# #                     embedding = embedding_0
# #                     score = score_0.detach()
#                     if self.queue is not None:
#                         if self.use_the_queue or not torch.all(self.queue[queue_i, :, -1]==0):
#                             self.use_the_queue=True
#                             score = torch.cat((torch.mm(self.queue[queue_i].t(), self.prototypes.weight.t()), score))
#                             self._dequeue_and_enqueue_swav(embedding, queue_i, pool) # 更新队列

#                     q = self.distributed_sinkhorn(score, epsilon, sinkhorn_iterations)[-bs_i:] #assign
            
#                 subloss = 0
#                 # 计算与其他标准增强样本的loss(其实就1个)
#                 for v in np.delete(range(nmb_crops[0]), crop_id):
#                     label = score_0[bs_i * v: bs_i * (v + 1)] / temperature
#                     subloss -= torch.mean(torch.sum(q * F.log_softmax(label, dim=1), dim=1))
#                 # 计算与其他非标准增强样本的loss
#                 for v in range(np.sum(nmb_crops)-nmb_crops[0]):
#                     label = score_1[bs_i * v: bs_i * (v + 1)] / temperature
#                     subloss = subloss - torch.mean(torch.sum(q * F.log_softmax(label, dim=1), dim=1))
#                 loss_i = loss_i + subloss / (sum(nmb_crops) - 1)
                
#             loss_i = loss_i / len(crops_for_assign) # 默认情况下crops_for_assign==[0,1]
#             loss_i.backward()
#             pdb.set_trace()
#             query_i_0.to('cpu')
#             query_i_1.to('cpu')
#             score_0.to('cpu')
#             score_1.to('cpu')
#             torch.cuda.empty_cache()
#             torch.cuda.empty_cache()
#             torch.cuda.empty_cache()
#             torch.cuda.empty_cache()
#             torch.cuda.empty_cache()

#         total_loss = total_loss + loss_i
        
#         # 由于query涉及到与client端网络和server端网络相关的两个计算图，因此在反向传播前先将query_0的梯度清零
#         if query_0.grad is not None:
#             query_0.grad.zero_() 
            
#         total_loss.backward()
#         # 默认情况下crops_for_assign==[0,1]
#         gradient[client_index] = query_0.grad.detach().clone() # get gradient, the -1 is important, since updates are added to the weights in cpp.
#             # query_0.grad.size() == torch.Size([32, 64, 224, 224])?
            
#         error = total_loss.detach().cpu().numpy() # error变量用于记录loss的数值

#         return error, gradient
    
    
    def compute_swav(self, client_embedding, crops_for_assign, nmb_crops, temperature, epsilon, sinkhorn_iterations, enqueue = True, pool = None):
        # compute函数用于server端网络在已有表征的基础上计算对比loss，并针对server端的网络反向传播计算server端网络的梯度。同时，将keys的最终表征pkey_out入队作为负例。
        # client_embedding是两个tensor组成的list
        
#         print(f"server-side prototypes：{self.model}")
        # normalize the prototypes
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)
        self.prototypes.to(self.device)
            
        # ============ swav loss ... ============
        query_0 = client_embedding[0] # 标准crop的tensor. [B*num_crops[0]*len(pool), args.K_dim]
        query_1 = client_embedding[1] # 其余crop的tensor. [B*num_crops[1]*len(pool), args.K_dim]
        query_0.requires_grad=True
        query_0.retain_grad()
        bs_i = int(query_0.size(0)/nmb_crops[0]) 
        
        # 计算两类增强样本的进一步表征
#         pdb.set_trace()
        embedding_0 = self.model(query_0.to(self.device, non_blocking=True))
        embedding_1 = self.model(query_1.to(self.device, non_blocking=True))
        
        # 用prototypes层处理，计算两类增强样本表征的分类scores。
        score_0 = self.prototypes(embedding_0) #用prototypes层处理，得到所有表征的分类scores。
        score_1 = self.prototypes(embedding_1)

        loss_i = 0 # 第i个client的loss
        for queue_i, crop_id in enumerate(crops_for_assign): #crop_id==0,1
            with torch.no_grad():
                embedding = embedding_0[bs_i * crop_id: bs_i * (crop_id + 1)]
                score = score_0[bs_i * crop_id: bs_i * (crop_id + 1)].detach()
                if self.queue is not None:
                    if self.use_the_queue or not torch.all(self.queue[queue_i, :, -1]==0):
                        self.use_the_queue=True
                        score = torch.cat((torch.mm(self.queue[queue_i].t(), self.prototypes.weight.t()), score))
                        self._dequeue_and_enqueue_swav(embedding, queue_i, pool) # 更新队列

                q = self.distributed_sinkhorn(score, epsilon, sinkhorn_iterations)[-bs_i:] #assign

            subloss = 0
            # 计算与其他标准增强样本的loss(其实就1个)
            for v in np.delete(range(nmb_crops[0]), crop_id):
                label = score_0[bs_i * v: bs_i * (v + 1)] / temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(label, dim=1), dim=1))
            # 计算与其他非标准增强样本的loss
            for v in range(sum(nmb_crops)-nmb_crops[0]):
                label = score_1[bs_i * v: bs_i * (v + 1)] / temperature
                subloss = subloss - torch.mean(torch.sum(q * F.log_softmax(label, dim=1), dim=1))
            loss_i = loss_i + subloss / (sum(nmb_crops) - 1)

        loss_i = loss_i / len(crops_for_assign) # 默认情况下crops_for_assign==[0,1]
        # 由于query涉及到与client端网络和server端网络相关的两个计算图，因此在反向传播前先将query_0的梯度清零
        if query_0.grad is not None:
            query_0.grad.zero_() 
        
        loss_i.backward()
        torch.cuda.empty_cache()
        # 默认情况下crops_for_assign==[0,1]
        gradient = query_0.grad.detach().clone() # get gradient, the -1 is important, since updates are added to the weights in cpp.
            # query_0.grad.size() == torch.Size([32, 64, 224, 224])?
            
        error = loss_i.detach().cpu().numpy() # error变量用于记录loss的数值

        return error, gradient
    
    def cuda(self, device):
        self.model.to(device)
        self.t_model.to(device)
    
    def cpu(self):
        self.model.cpu()
        self.t_model.cpu()

class create_sflmococlient_instance(create_base_instance):
    def __init__(self, model) -> None:
        super().__init__(model) # 传入的model是client-side model
#         self.output = None
        self.output = []
        self.t_model = copy.deepcopy(model) 
        for param_t in self.t_model.parameters():
            param_t.requires_grad = False  # not update by gradient
    
    def __call__(self, input):
        return self.forward(input)

    def forward(self, input): # return a detached one.
#         print(f"输入数据list的长度{len(input)}")
        if self.output is None:
            self.output = []
        if not isinstance(input, list):
            input = [input]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in input]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
#             _out = self.model(torch.cat(input[start_idx: end_idx]).cuda(non_blocking=True)) 
            _out = self.model(torch.cat(input[start_idx: end_idx])).detach()
            self.output.append(_out)
#             if start_idx == 0:
#                 self.output = _out
#             else:
#                 self.output = torch.cat((self.output, _out)) 
#                 # 这里直接拼接会报错。因为不像swav的output是经过了整个model的结果，本实验的当前model只是client-side model，即一个conv_gn，其输出tensor的最后两维依然和size有关，所以无法拼接。
            start_idx = end_idx
#         self.output = self.model(input) # 计算input的表征
#         self.update_moving_average()

        # self.output是个list，其中包含了len(self.size_crops)==2个tensor。第一个tensor包含nmb_crops[0]==2个标准增强样本的表征，第二个tensor包含nmb_crops[1]==6个增强样本的表征
#         return self.output.detach() 
        return self.output  # .detach()操作放到compute_swav中执行

    def backward(self, external_grad):
        if self.output is not None:
#             print(f"self.output[0]的size为:{self.output[0].size()}")
            self.output[0].backward(gradient=external_grad)
            self.output = None
            
    @torch.no_grad()
    def update_moving_average(self): # client端也是使用momentum encoder计算keys
        tau = 0.99 # default value in moco
        for online, target in zip(self.model.parameters(), self.t_model.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
    
    def cuda(self, device):
        self.model.to(device)
        self.t_model.to(device)
    
    def cpu(self):
        self.model.cpu()
        self.t_model.cpu()