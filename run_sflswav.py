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
from functions.sflswav_functions import sflmoco_simulator
from functions.sfl_functions import client_backward, loss_based_status, client_backward_swav
from functions.attack_functions import MIA_attacker, MIA_simulator
from Dali.get_dali_dataloader import get_cifar10_dali, get_cifar10_Dali_loader, get_cifar10_dali_DDP
from Dali.load_cifar10_data import load_cifar10
from torch.cuda import amp
import gc
import pdb
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
VERBOSE = False
#get default args
args = get_sfl_args()
set_deterministic(args.seed)

# print(f"print args:\n {args}")
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3,4,5"

'''Preparing'''
#get data
if args.is_distributed:
    local_rank = int(os.environ["LOCAL_RANK"]) 
    print(f"local_rank: {local_rank}\n")
    world_size = int(os.environ["WORLD_SIZE"]) 
    print(f"world_size: {world_size}")

    # 新增：DDP backend初始化
    torch.cuda.set_device(local_rank) # 设置device
    dist.init_process_group(backend='nccl')  # 初始化多进程. nccl是GPU设备上最快、最推荐的后端
    
    # load data
    train_loader, traindata_cls_counts, mem_loader, test_loader = get_cifar10_dali_DDP(
        size_crops=args.size_crops, nmb_crops=args.nmb_crops, min_scale_crops=args.min_scale_crops, max_scale_crops=args.max_scale_crops, 
        batch_size=args.batch_size, num_workers=args.num_workers, num_client = args.num_client, 
        data_proportion = args.data_proportion, noniid_ratio =args.noniid_ratio, 
        pairloader_option = args.pairloader_option, 
        partition = args.partition, partition_beta = 0.4, hetero = args.hetero, 
        path_to_data = args.data_dir, world_size = world_size, local_rank=local_rank)

if args.use_dali is True:
    train_loader, traindata_cls_counts, mem_loader, test_loader = get_cifar10_dali(
        size_crops=args.size_crops, nmb_crops=args.nmb_crops, min_scale_crops=args.min_scale_crops, max_scale_crops=args.max_scale_crops, 
        batch_size=args.batch_size, num_workers=args.num_workers, num_client = args.num_client, 
        data_proportion = args.data_proportion, noniid_ratio =args.noniid_ratio, 
        pairloader_option = args.pairloader_option, 
        partition = args.partition, partition_beta = 0.4, hetero = args.hetero, 
        path_to_data = args.data_dir)
    # 之后遍历train_loader要先把数据的size还原
    world_size = 1
else:
    create_dataset = getattr(datasets, f"get_{args.dataset}")
    train_loader, traindata_cls_counts, mem_loader, test_loader =create_dataset(
        args.size_crops, 
        args.nmb_crops, 
        args.min_scale_crops, 
        args.max_scale_crops, 
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        shuffle=True, 
        num_client = args.num_client, 
        data_proportion = args.data_proportion, 
        noniid_ratio = args.noniid_ratio, 
        augmentation_option = True, 
        pairloader_option = args.pairloader_option, 
        aug_type = args.aug_type,
        hetero = args.hetero, 
        hetero_string = args.hetero_string)
    world_size = 1
# datasets.py
# train_loader ==  get_cifar10_pairloader （train_loader作为对比学习的训练集）
# mem_loader == get_cifar10_trainloader(128, num_workers, False, path_to_data = path_to_data)
# （mem_loader 中的数据在KNN中作为基准样本点，在线性分类中作为训练线性分类层的训练集）
# test_loader == get_cifar10_testloader(128, num_workers, False, path_to_data)
# （test_loader 中的数据用于最终的线性分类测试和半监督测试）
# pdb.set_trace()
# for data in train_loader[0]: 
#     print(data.size())
#train_loader里包含的是若干个clients的dataloader. 每个dataloader中包含增强样本数量个tensor,每个tensor的维度为[batch_size, 3, size_crop, size_crop]
num_batch = len(train_loader[0])*world_size # 将第0个client拥有数据的数量作为server-side model的训练epoch数。（即整个模型训练arg.num_epoch轮，每轮中server-side model训练num_batch个batch。每个batch中所有client计算自己数据的表征并将表征在server端聚合）

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
                           size_crops = args.size_crops,
                           nmb_crops = args.nmb_crops,
                           group_norm=True, 
                           input_size= args.data_size, 
                           adds_bottleneck=args.adds_bottleneck,
                           bottleneck_option=args.bottleneck_option, 
                           c_residual = args.c_residual, 
                           WS = args.WS,
                           device = args.device)

if args.mlp: # 只有moco V1的mlp是False
    if args.moco_version == "largeV2": # This one uses a larger classifier, same as in Zhuang et al. Divergence-aware paper
        classifier_list = [nn.Linear(output_dim * global_model.expansion, 4096),
                        nn.BatchNorm1d(4096),
                        nn.ReLU(True),
                        nn.Linear(4096, args.K_dim)] #resnet的output_dim==512
    elif "V2" in args.moco_version:
        classifier_list = [nn.Linear(output_dim * global_model.expansion, args.K_dim * global_model.expansion),
                        nn.ReLU(True),
                        nn.Linear(args.K_dim * global_model.expansion, args.K_dim)]
        # K_dim在configs/__init__.py中定义，是随模型的变化设定的。当前程序==1024。
    else:
        raise("Unknown version! Please specify the classifier.")
    
    global_model.classifier = nn.Sequential(*classifier_list)
    global_model.classifier.apply(init_weights)
# print(f"global_model的层\n{global_model}")
global_model.merge_classifier_cloud() # 给模型加上mlp层（global_model的结果依然是对Input的表征）
# 用于训练的整个线性层包括了MLP层

if args.is_distributed:
    device = torch.device("cuda", local_rank) # 获取device，之后的模型和张量都.to(device)
else:
    device = args.device
    
#get loss function
criterion = nn.CrossEntropyLoss().to(device)

#initialize sfl
sfl = sflmoco_simulator(global_model, criterion, train_loader, test_loader, device, args)

sfl.log(args)

if args.use_fp16:
    scaler = amp.GradScaler()
else:
    scaler = None

# '''Initialze with ResSFL resilient model '''  # resilient model是什么？
# if args.initialze_path != "None":
#     sfl.log("Load from resilient model, train with client LR of {}".format(args.c_lr))
#     sfl.load_model_from_path(args.initialze_path, load_client = True, load_server = args.load_server)
#     args.attack = True


sfl.cuda(device)
# ###原来的sfl####
# if args.cutlayer > 1: # 为什么设置了cutlayer就要把sfl加载到cuda上？
# #     sfl.cuda() # sfl加载到cuda，就是把sfl的server-side model和client-side model都加载到cuda
#     sfl.cuda(args.device)
# else:
#     sfl.cpu()
# sfl.s_instance.cuda(args.device)

# '''ResSFL training''' 
# if args.enable_ressfl: # 这一部分暂时没用
#     sfl.log(f"Enable ResSFL fine-tuning: arch-{args.MIA_arch}-alpha-{args.ressfl_alpha}-ssim-{args.ressfl_target_ssim}")
#     ressfl = MIA_simulator(sfl.model, args, args.MIA_arch)
#     ressfl.cuda()
#     ressfl.cuda(args.device)
#     args.attack = True


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
#         avg_gan_train_loss = 0.0
#         avg_gan_eval_loss = 0.0
        for batch in range(num_batch):
            sfl.optimizer_zero_grads()
            iteration = epoch * num_batch + batch
            
            if loss_status.status == "A" or loss_status.status == "B":
                # 如果当前状态为A/B级,即两次epoch之间的训练loss相差比较大,则使用更新后的client-side网络重新计算query表征; 否则表示以前的表征已经训练得比较好了,则直接使用上一轮计算得到的query表征.
                client_embedding_list = [None for _ in range(len(pool))] 
                # 用于存放每个client的所有hidden_query
                scores_list = [None for _ in range(len(pool))]
                gradient_dict = {i:None for i in range(len(pool))}
                bs = []
                #client forward
                for client_index, client_id in enumerate(pool): # if distributed, this can be parallelly done.
                    images = sfl.next_swavdata_batch(client_id, args.use_dali)  # images是个list，其中包含2+6个crops
#                     print(f"第{epoch}轮第{batch}个batch第{client_id}号client读取数据条数为{len(images)}")
                    if args.use_dali:
                        data = images['data']
#                         images_dict = {i: None for i in range(sum(args.nmb_crops))}
                        images_list = [None for _ in range(sum(args.nmb_crops))]
                        ptr = 0
                        column_add_pre = 0
                        for crop_id in range(len(args.nmb_crops)): # 0,1
                            for num_id in range(args.nmb_crops[crop_id]): #NMB_CROPS[0]==2
                                column_add = column_add_pre+args.size_crops[crop_id] * args.size_crops[crop_id]
#                                 pdb.set_trace()
                                image = images['data'][:,:, :, column_add_pre: column_add]
#                                 print(f"size of images_dict[{ptr}]: {image.size()}")
                                images_list[ptr] = (image.reshape([args.batch_size, 3, args.size_crops[crop_id], -1])).to('cpu')
                                column_add_pre = column_add
                    #             print(f"column_add_pre=:{column_add_pre}")
                    #             print(f"column_add=:{column_add}")
#                                 print(f"size of images_list[{ptr}]: {images_list[ptr].size()}")
                                ptr+=1
                        images = images_list
#                     pdb.set_trace()
#                     if args.cutlayer > 1:
                    for image_id in range(len(images)):
                        images[image_id] = images[image_id].to(args.device)
#                     if batch==1:
#                     pdb.set_trace()
                    embedding = sfl.c_instance_list[client_id](images)# (client-side model计算表征)
                    # embedding是个list，其中包含了len(self.size_crops)==2个tensor，其中第一个tensor包含nmb_crops[0]==2个标准增强样本的表征，第二个tensor包含nmb_crops[1]==6个增强样本的表征
                    #.detach操作在sflswav_functions.py-->create_sflmococlient_instance-->forward函数中做过了
                    # 使用client-side部分对aug1进行表征
                    client_embedding_list[client_index] = embedding 
                    bs.append(int(embedding[0].size(0)/args.nmb_crops[0]))
                    
#                 stack_hidden_query = torch.cat(hidden_query_list, dim = 0)#将所有client的表征拼接起来
                # stack_hidden_pkey：(num_client*batch_size, hidden_size, input_size, input_size)

#                 if args.loss_threshold > 0.0:
#                     torch.save(stack_hidden_query, f"replay_tensors/stack_hidden_query_{batch}.pt")
#                     torch.save(stack_hidden_pkey, f"replay_tensors/stack_hidden_pkey_{batch}.pt")
#             else: # 当loss变化比较小，则使用client-side models在上一个epoch中计算得到的表征
#                 stack_hidden_query = torch.load(f"replay_tensors/stack_hidden_query_{shuffle_map[batch]}.pt")
#                 stack_hidden_pkey = torch.load(f"replay_tensors/stack_hidden_pkey_{shuffle_map[batch]}.pt")
            
#             stack_hidden_query = stack_hidden_query.to(args.device)

                sfl.s_optimizer.zero_grad()
                #server compute
    #             loss, gradient, accu = sfl.s_instance.compute(hidden_embedding_list, scores_list, pool = pool) # loss是对比loss，gradient是所有query的梯度, accu是对比acc

                 #注意：此时两个list均在cpu中
#                 pdb.set_trace()
                loss, gradient= sfl.s_instance.compute_swav(
                    client_embedding_list, 
                    crops_for_assign = args.crops_for_assign, 
                    nmb_crops = args.nmb_crops, 
                    temperature = args.temperature, 
                    epsilon = args.epsilon, 
                    sinkhorn_iterations = args.sinkhorn_iterations, 
                    pool = pool,
                    use_fp16 = args.use_fp16,
                    scaler = scaler) 
                
                    
#                 print(f"epoch:{epoch}, batch:{batch}, client:{client_index}, loss:{loss}")
                print(f"epoch:{epoch}, batch:{batch}, loss:{loss}")

                if iteration < args.freeze_prototypes_niters:
                    for name, p in model.named_parameters():
                        if "prototypes" in name:
                            p.grad = None

                if args.use_fp16:
                    scaler.step(sfl.s_optimizer)
#                     scaler.update()
                else:
                    sfl.s_optimizer.step() # with reduced step, to simulate a large batch size.

                if VERBOSE and (batch % 50 == 0 or batch == num_batch - 1):
                    sfl.log(f"epoch {epoch} batch {batch}, loss {loss}")
                avg_loss += loss
    #             avg_accu += accu

                # distribute gradients to clients
                start_index = 0
#                 pdb.set_trace()
                for i in range(len(pool)):
                    gradient_i = gradient[start_index : start_index + bs[i] * len(args.crops_for_assign)]
#                     if args.cutlayer <= 1:
#                         gradient_i = gradient_i.cpu()
                    gradient_dict[i] = gradient_i
                    start_index = start_index+bs[i] * len(args.crops_for_assign)
                    

            if loss_status.status == "A": # loss大于某个阈值的时候才更新client-side models
                # Initialize clients' queue, to store partial gradients
                #client backward
#                 pdb.set_trace()
                client_backward_swav(sfl, pool, gradient_dict, use_fp16 = args.use_fp16, scaler = scaler) # 各client以自己的gradient进行反向传播
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

#         avg_accu = avg_accu / num_batch # avg_accu是一个epoch中所有batch的对比acc的batch平均
        avg_loss = avg_loss / num_batch # avg_loss是一个epoch中所有batch的对比loss的batch平均
        if args.enable_ressfl:
            avg_gan_train_loss = avg_gan_train_loss / num_batch / len(pool)
            avg_gan_eval_loss = avg_gan_eval_loss / num_batch / len(pool)
        
        loss_status.record_loss(epoch, avg_loss) # 更新loss的状态
        
        knn_val_acc = sfl.knn_eval(memloader=mem_loader, use_dali = args.use_dali) # 每个epoch计算一次knn_acc。
        # mem_loader被用于KNN分类中的已知类别的样本点集合（是一个DataLoader的list）
#         if args.cutlayer <= 1:
#             sfl.c_instance_list[0].cpu()
        if knn_val_acc > knn_accu_max:  # 用knn_accu_max保存最优acc，并保存knn准确率最高时的最优模型(包括server-side model和第0个client-side model)。
            knn_accu_max = knn_val_acc
            sfl.save_model(epoch, is_best = True) # (base_funtions.py)
        epoch_logging_msg = f"epoch:{epoch}, knn_val_accu: {knn_val_acc:.2f}, contrast_loss: {avg_loss:.2f}, contrast_acc: {avg_accu:.2f}" # contrast_acc：正例对的logits比负例对的logits高则正确
        epoch_logging_msg = f"epoch:{epoch}, knn_val_accu: {knn_val_acc:.2f}, contrast_loss: {avg_loss:.2f}" # contrast_acc：正例对的logits比负例对的logits高则正确
        
        if args.enable_ressfl:
            epoch_logging_msg += f", gan_train_loss: {avg_gan_train_loss:.2f}, gan_eval_loss: {avg_gan_eval_loss:.2f}"
        
        sfl.log(epoch_logging_msg)
        gc.collect()
if args.loss_threshold > 0.0: #如果使用降级策略
    saving = loss_status.epoch_recording["C"] + loss_status.epoch_recording["B"]/2
    sfl.log(f"Communiation saving: {saving} / {args.num_epoch}") #这是在干什么？


'''Testing'''
# sfl.load_model() # load model that has the lowest contrastive loss. 
# 加载所有epoch中KNN准确率最高的模型，并使用该模型进行下一步的测试。
# finally, do a thorough evaluation.
val_acc = sfl.knn_eval(memloader=mem_loader, use_dali = args.use_dali)
sfl.log(f"final knn evaluation accuracy is {val_acc:.2f}")

if args.use_dali is True:
    eval_loader = get_cifar10_Dali_loader([test_data], [test_targets], 1, 128, 1.0, args.num_workers, train=False)
    val_acc = sfl.linear_eval(eval_loader, 100, use_dali = args.use_dali)
    sfl.log(f"final linear-probe evaluation accuracy is {val_acc:.2f}")

    eval_loader = get_cifar10_Dali_loader([test_data], [test_targets], 1, 128, 0.1, args.num_workers, train=False)
    val_acc = sfl.semisupervise_eval(eval_loader, 100, use_dali = args.use_dali)
    sfl.log(f"final semi-supervised evaluation accuracy with 10% data is {val_acc:.2f}")

    eval_loader = get_cifar10_Dali_loader([test_data], [test_targets], 1, 128, 0.01, args.num_workers, train=False)
    val_acc = sfl.semisupervise_eval(eval_loader, 100, use_dali = args.use_dali)
    sfl.log(f"final semi-supervised evaluation accuracy with 1% data is {val_acc:.2f}")
else:
    create_train_dataset = getattr(datasets, f"get_{args.dataset}_trainloader")
    # create_train_dataset(128, args.num_workers, False, num_client = 1, data_portion = 1.0, noniid_ratio =1.0, augmentation_option =False)
    # 下面3个eval_loader的区别在于data_portion不同。
    eval_loader = create_train_dataset(128, args.num_workers, False, 1, 1.0, 1.0, False)
    val_acc = sfl.linear_eval(eval_loader, 100, use_dali = args.use_dali)
    sfl.log(f"final linear-probe evaluation accuracy is {val_acc:.2f}")

    eval_loader = create_train_dataset(128, args.num_workers, False, 1, 0.1, 1.0, False)
    val_acc = sfl.semisupervise_eval(eval_loader, 100, use_dali = args.use_dali)
    sfl.log(f"final semi-supervised evaluation accuracy with 10% data is {val_acc:.2f}")

    eval_loader = create_train_dataset(128, args.num_workers, False, 1, 0.01, 1.0, False)
    val_acc = sfl.semisupervise_eval(eval_loader, 100, use_dali = args.use_dali)
    sfl.log(f"final semi-supervised evaluation accuracy with 1% data is {val_acc:.2f}")

if args.attack:
    '''Evaluate Privacy'''
    if args.resume:
        sfl.load_model() # load model that has the lowest contrastive loss.
    val_acc = sfl.knn_eval(memloader=mem_loader, use_dali=args.use_dali)
    sfl.log(f"final knn evaluation accuracy is {val_acc:.2f}")
    MIA = MIA_attacker(sfl.model, train_loader, args, "res_normN4C64")
    MIA.MIA_attack()