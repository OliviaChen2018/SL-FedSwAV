# %%
'''
MocoSFL
'''
from cmath import inf
import datasets
from configs import get_sfl_args, set_deterministic
import torch
import torch.nn as nn
import numpy as np
from models import resnet
from models import vgg
from models import mobilenetv2
from models.resnet import init_weights
from functions.sflmoco_functions import sflmoco_simulator
from functions.sfl_functions import client_backward, loss_based_status
from functions.attack_functions import MIA_attacker, MIA_simulator
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
import os
VERBOSE = False
#get default args
args = get_sfl_args()
set_deterministic(args.seed)

if args.is_distributed :
    os.environ['CUDA_VISIBLE_DEVICES'] = "2,3,4,5" # SflMoco
    local_rank = int(os.environ["LOCAL_RANK"]) 
    print(f"local_rank: {local_rank}\n")
    world_size = int(os.environ["WORLD_SIZE"]) 
    print(f"world_size: {world_size}")

    # 新增：DDP backend初始化
    torch.cuda.set_device(local_rank) # 设置device
    dist.init_process_group(backend='nccl')  # 初始化多进程. nccl是GPU设备上最快、最推荐的后端
    
    device = torch.device("cuda", local_rank) # 获取device，之后的模型和张量都.to(device)
else:
    device = args.device

'''Preparing'''
#get data
create_dataset = getattr(datasets, f"get_{args.dataset}")
train_loader, traindata_cls_counts, mem_loader, test_loader =create_dataset(
    batch_size=args.batch_size,
    num_workers=args.num_workers, 
    shuffle=True, 
    num_client = args.num_client, 
    data_proportion = args.data_proportion, 
    noniid_ratio = args.noniid_ratio, 
    augmentation_option = True, 
    pairloader_option = args.pairloader_option, 
    hetero = args.hetero, 
    hetero_string = args.hetero_string,
    is_distributed = args.is_distributed)
# datasets.py
# train_loader ==  get_cifar10_pairloader （train_loader作为对比学习的训练集）
# mem_loader == get_cifar10_trainloader(128, num_workers, False, path_to_data = path_to_data)
# （mem_loader 中的数据在KNN中作为基准样本点，在线性分类中作为训练线性分类层的训练集）
# test_loader == get_cifar10_testloader(128, num_workers, False, path_to_data)
# （test_loader 中的数据用于最终的线性分类测试和半监督测试）

num_batch = len(train_loader[0]) # 将第0个client拥有数据的数量作为server-side model的训练epoch数。（即整个模型训练arg.num_epoch轮，每轮中server-side model训练num_batch个batch。每个batch中所有client计算自己数据的表征并将表征在server端聚合）

# resnet, vgg, MobileNetV2分别是./models/中的三个model文件。
# e.g. args.arch== 'ResNet18'，
# 则getattr(resnet, args.arch): 在from models import resnet文件中找到ResNet18函数, 作为实体返回

if "ResNet" in args.arch or "resnet" in args.arch:
    if "resnet" in args.arch:
        args.arch = "ResNet" + args.arch.split("resnet")[-1]
    create_arch = getattr(resnet, args.arch) # 例如create_arch==ResNet18
    output_dim = 512 # 分类层之前的层输出的特征维数为512
elif "vgg" in args.arch:
    create_arch =  getattr(vgg, args.arch)
    output_dim = 512
elif "MobileNetV2" in args.arch:
    create_arch =  getattr(mobilenetv2, args.arch)
    output_dim = 1280
#get model - use a larger classifier, as in Zhuang et al. Divergence-aware paper
global_model = create_arch(cutting_layer=args.cutlayer, 
                           num_client = args.num_client, 
                           num_class=args.K_dim, 
                           nmb_prototypes = args.nmb_prototypes,
                           group_norm=True, 
                           input_size= args.data_size, 
                           adds_bottleneck=args.adds_bottleneck,
                           bottleneck_option=args.bottleneck_option, 
                           c_residual = args.c_residual, 
                           WS = args.WS, 
                           device = device)

if args.mlp: # 只有moco V1的mlp是False
    if args.moco_version == "largeV2": # This one uses a larger classifier, same as in Zhuang et al. Divergence-aware paper
        classifier_list = [nn.Linear(output_dim * global_model.expansion, 4096),
                        nn.BatchNorm1d(4096),
                        nn.ReLU(True),
                        nn.Linear(4096, args.K_dim)]
    elif "V2" in args.moco_version:
        classifier_list = [nn.Linear(output_dim * global_model.expansion, args.K_dim * global_model.expansion),
                        nn.ReLU(True),
                        nn.Linear(args.K_dim * global_model.expansion, args.K_dim)]
    else:
        raise("Unknown version! Please specify the classifier.")
    
    global_model.classifier = nn.Sequential(*classifier_list)
    global_model.classifier.apply(init_weights)
global_model.merge_classifier_cloud() # 给模型加上mlp层（global_model的结果依然是对Input的表征）
# 用于训练的整个线性层包括了MLP层

#get loss function
criterion = nn.CrossEntropyLoss().to(args.device)

#initialize sfl
if args.is_distributed:
    sfl = sflmoco_simulator(global_model, criterion, train_loader, test_loader, args, device=device, local_rank=local_rank, is_distributed=args.is_distributed)
else:
    sfl = sflmoco_simulator(global_model, criterion, train_loader, test_loader, args)

# if args.cutlayer > 1: # 为什么设置了cutlayer就要把sfl加载到cuda上？
# #     sfl.cuda() # sfl加载到cuda，就是把sfl的server-side model和client-side model都加载到cuda
#     sfl.cuda(args.device)
# else:
#     sfl.cpu()
# sfl.s_instance.cuda(args.device)

sfl.log(f'Data statistics: {str(traindata_cls_counts)}')
    
'''Training'''
if not args.resume: # 模型从头训练(而不是resume from checkpoint)
    sfl.log(f"SFL-Moco-microbatch (Moco-{args.moco_version}, Hetero: {args.hetero}, Sample_Ratio: {args.client_sample_ratio}) Train on {args.dataset} with cutlayer {args.cutlayer} and {args.num_client} clients with {args.noniid_ratio}-data-distribution: total epochs: {args.num_epoch}, total number of batches for each client is {num_batch}")
    
    sfl.train()
    #Training scripts (SFL-V1 style)
    knn_accu_max = 0.0

    #heterogeneous resources setting
    if args.hetero:
        pass
#         sfl.log(f"Hetero setting: {args.hetero_string}")
#         rich_clients = int(float(args.hetero_string.split("|")[0].split("_")[0]) * args.num_client)
#         rich_clients_batch_size = int(float(args.hetero_string.split("|")[1]) * args.batch_size)
    
    loss_status = loss_based_status(loss_threshold = args.loss_threshold)
    
    for epoch in range(1, args.num_epoch + 1):
        
        if args.loss_threshold > 0.0:
            print(f"loss_status: {loss_status.status}")

        if loss_status.status == "C":
            shuffle_map = np.random.permutation(range(num_batch)) # shuffle map for communicate

        if args.client_sample_ratio == 1.0:
            pool = range(args.num_client) 
        else: # pool list存放本轮参与训练的clients的下标
            pool = np.random.choice(range(args.num_client), int(args.client_sample_ratio * args.num_client), replace=False) # 10 out of 1000
        
        avg_loss = 0.0
        avg_accu = 0.0
        avg_gan_train_loss = 0.0
        avg_gan_eval_loss = 0.0
        for batch in range(num_batch):
            sfl.optimizer_zero_grads()

            if loss_status.status == "A" or loss_status.status == "B":
                # 如果当前状态为A/B级,即两次epoch之间的训练loss相差比较大,则使用更新后的client-side网络重新计算query表征; 否则表示以前的表征已经训练得比较好了,则直接使用上一轮计算得到的query表征.
                hidden_query_list = [None for _ in range(len(pool))] 
                # 用于存放每个client的所有hidden_query
                hidden_pkey_list = [None for _ in range(len(pool))]
                query_num = [0 for _ in range(len(pool))] #(我加的，用于统计每个client有几条数据)

                #client forward
                for i, client_id in enumerate(pool): # if distributed, this can be parallelly done.
                    query, pkey = sfl.next_data_batch(client_id) 
                    #获得第client_id个client的train_data，即augmented images正例对的两个batch
                    query_num[i] = query.size(0)
#                     if args.cutlayer > 1:
                    query = query.to(device)
                    pkey = pkey.to(device)
                    hidden_query = sfl.c_instance_list[client_id](query)# pass to online  
                    #是不是应该.detach()啊？client的forward函数的返回值已经做了detach了。
                    # 使用client-side部分对aug1进行表征
                    hidden_query_list[i] = hidden_query #将aug1(query)的表征用一个list保存起来
                    with torch.no_grad():
                        # pass to target 
                        hidden_pkey = sfl.c_instance_list[client_id].t_model(pkey).detach()  
                        # self.t_model = copy.deepcopy(model)
                        # 在MoCo中，生成key的encoder进行动量更新(而不通过梯度反向传播),因此要使用一个深拷贝的模型，并.detach切断梯度计算链，再计算aug2(key)的表征。
                    hidden_pkey_list[i] = hidden_pkey # 将aug2(key)的表征用一个list保存起来
                    # hidden_pkey_list: [tensor, tensor, ...,tensor]，其中tensor.size()==[batch_size, hidden_size, input_size, input_size]

                stack_hidden_query = torch.cat(hidden_query_list, dim = 0) 
                stack_hidden_pkey = torch.cat(hidden_pkey_list, dim = 0) # 将所有client的表征拼接起来
                # stack_hidden_pkey：(num_client*batch_size, hidden_size, input_size, input_size)

                if args.loss_threshold > 0.0:
                    torch.save(stack_hidden_query, f"replay_tensors/stack_hidden_query_{batch}.pt")
                    torch.save(stack_hidden_pkey, f"replay_tensors/stack_hidden_pkey_{batch}.pt")
            else: # 当loss变化比较小，则使用client-side models在上一个epoch中计算得到的表征
                stack_hidden_query = torch.load(f"replay_tensors/stack_hidden_query_{shuffle_map[batch]}.pt")
                stack_hidden_pkey = torch.load(f"replay_tensors/stack_hidden_pkey_{shuffle_map[batch]}.pt")
            
            stack_hidden_query = stack_hidden_query.to(device)
            stack_hidden_pkey = stack_hidden_pkey.to(device)
#             print(f"stack_hidden_query的device: {device}")

            sfl.s_optimizer.zero_grad()
    
            #server compute
#             loss, gradient, accu = sfl.s_instance.compute(stack_hidden_query, stack_hidden_pkey, pool = pool, world_size = world_size) # loss是对比loss，gradient是所有query的梯度, accu是对比acc

            loss, gradient, accu = sfl.s_instance.compute_simco(stack_hidden_query, stack_hidden_pkey, pool = pool, world_size = world_size)

            sfl.s_optimizer.step() # with reduced step, to simulate a large batch size.

            if VERBOSE and (batch% 50 == 0 or batch == num_batch - 1):
                sfl.log(f"epoch {epoch} batch {batch}, loss {loss}")
            avg_loss += loss
            avg_accu += accu

            # distribute gradients to clients
#             if args.cutlayer <= 1:
#                 gradient = gradient.cpu()

            if loss_status.status == "A": # loss大于某个阈值的时候才更新client-side models
                # Initialize clients' queue, to store partial gradients
                gradient_dict = {key: [] for key in range(len(pool))} # 用于存放每个client的gradient
                
                # 如果使用Dirichlet分布划分client数据，则需要在query计算的时候记录client有几条数据，以便于这里gradient的分配。

                # 将梯度返回给各client
                if not args.hetero:
                    for j in range(len(pool)):
                        gradient_dict[j] = gradient[j*args.batch_size:(j+1)*args.batch_size, :]
                        
                else:
#                     start_grad_idx = 0
                    query_index = torch.tensor(query_num).cumsum(dim=0)
                    for j in range(len(pool)):
                        if j==0:
                            gradient_dict[j] = gradient[0:query_index[j], :]
                        else:
                            gradient_dict[j] = gradient[query_index[j-1]:query_index[j], :]
#                         gradient_dict[j].size()==[bs, 64, input_size, input_size]
#                         if (pool[j]) < rich_clients: # if client is rich. Implement hetero backward.
#                             gradient_dict[j] = gradient[start_grad_idx: start_grad_idx + rich_clients_batch_size]
#                             start_grad_idx += rich_clients_batch_size
#                         else:
#                             gradient_dict[j] = gradient[start_grad_idx: start_grad_idx + args.batch_size]
#                             start_grad_idx += args.batch_size

                #client backward
                client_backward(sfl, pool, gradient_dict) # 各client以自己的gradient进行反向传播
            else:
                # (optional) step client scheduler (lower its LR)
                pass

            gc.collect() # 这里只是为了避免出现内存泄露

            if batch == num_batch - 1 or (batch % (num_batch//args.avg_freq) == (num_batch//args.avg_freq) - 1):
                # sync client-side models 
                # num_batch==len(train_loader[0])，即训练集中batch的数量。
                divergence_list = sfl.fedavg(pool, divergence_aware = args.divergence_aware, divergence_measure = args.divergence_measure)
                # 每多少步，对client-side models进行divergence-aware的动量更新。
                # divergence_list中存放的是每个client-side model参数与参数均值之间的divergence。
                
                if divergence_list is not None:
                    sfl.log(f"divergence mean: {np.mean(divergence_list)}, std: {np.std(divergence_list)} and detailed_list: {divergence_list}")
                    
        sfl.s_scheduler.step()

        avg_accu = avg_accu / num_batch # avg_accu是一个epoch中所有batch的对比acc的batch平均
        avg_loss = avg_loss / num_batch # avg_loss是一个epoch中所有batch的对比loss的batch平均
        
        loss_status.record_loss(epoch, avg_loss) # 更新loss的状态
        
        knn_val_acc = sfl.knn_eval(memloader=mem_loader) # 每个epoch计算一次knn_acc。
        # mem_loader被用于KNN分类中的已知类别的样本点集合（是一个DataLoader的list）
#         if args.cutlayer <= 1:
#             sfl.c_instance_list[0].cpu()
        if knn_val_acc > knn_accu_max:  # 用knn_accu_max保存最优acc，并保存knn准确率最高时的最优模型(包括server-side model和第0个client-side model)。
            knn_accu_max = knn_val_acc
            sfl.save_model(epoch, is_best = True) # (base_funtions.py)
        epoch_logging_msg = f"epoch:{epoch}, knn_val_accu: {knn_val_acc:.2f}, contrast_loss: {avg_loss:.2f}, contrast_acc: {avg_accu:.2f}" # contrast_acc：正例对的logits比负例对的logits高则正确
        
        sfl.log(epoch_logging_msg)
        gc.collect()
if args.loss_threshold > 0.0: #如果使用降级策略
    saving = loss_status.epoch_recording["C"] + loss_status.epoch_recording["B"]/2
    sfl.log(f"Communiation saving: {saving} / {args.num_epoch}") #这是在干什么？


'''Testing'''
sfl.load_model() # load model that has the lowest contrastive loss. 
# 加载所有epoch中KNN准确率最高的模型，并使用该模型进行下一步的测试。
# finally, do a thorough evaluation.
val_acc = sfl.knn_eval(memloader=mem_loader)
sfl.log(f"final knn evaluation accuracy is {val_acc:.2f}")

create_train_dataset = getattr(datasets, f"get_{args.dataset}_trainloader")

# create_train_dataset(128, args.num_workers, False, num_client = 1, data_portion = 1.0, noniid_ratio =1.0, augmentation_option =False)
# 下面3个eval_loader的区别在于data_portion不同。
eval_loader = create_train_dataset(128, args.num_workers, False, 1, 1.0, 1.0, False)
val_acc = sfl.linear_eval(eval_loader, 100)
sfl.log(f"final linear-probe evaluation accuracy is {val_acc:.2f}")

eval_loader = create_train_dataset(128, args.num_workers, False, 1, 0.1, 1.0, False)
val_acc = sfl.semisupervise_eval(eval_loader, 100)
sfl.log(f"final semi-supervised evaluation accuracy with 10% data is {val_acc:.2f}")

eval_loader = create_train_dataset(128, args.num_workers, False, 1, 0.01, 1.0, False)
val_acc = sfl.semisupervise_eval(eval_loader, 100)
sfl.log(f"final semi-supervised evaluation accuracy with 1% data is {val_acc:.2f}")

if args.attack:
    '''Evaluate Privacy'''
    if args.resume:
        sfl.load_model() # load model that has the lowest contrastive loss.
    val_acc = sfl.knn_eval(memloader=mem_loader)
    sfl.log(f"final knn evaluation accuracy is {val_acc:.2f}")
    MIA = MIA_attacker(sfl.model, train_loader, args, "res_normN4C64")
    MIA.MIA_attack()