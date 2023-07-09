

def get_sfl_args():
    parser = argparse.ArgumentParser()
    
    # training specific args
    parser.add_argument('--dataset', type=str, default='cifar10', help='choose from cifar10, cifar100, imagenet')
    parser.add_argument('--num_class', type=int, default=10, help="number of classes: N")
    parser.add_argument('--data_size', type=int, default=32, help="dimension size of input data(是干啥的？)")
    
    # Split Learning Setting (Basic)
    
    # Split Learning Setting (Advanced, including optimization)
#     parser.add_argument('--data_proportion', type=float, default=1.0, help="Use subset of iid data，只使用原始数据集的一部分作为训练集。taressfl中使用")
    
    # Split Learning Non-IID specific Setting (Advanced)
    
    # Moco setting
    parser.add_argument('--model_version', type=str, default="swav", help="moco_version: V1, smallV2, V2, largeV2, swav")
    parser.add_argument('--pairloader_option', type=str, default="None", help="set a pairloader option (results in augmentation differences), only enable it in contrastive learning, choice: mocov1, mocov2, swav")
    
    # Moco-V2 setting
    
    
    dataset_name_list = ["cifar10", "cifar100", "imagenet", "svhn", "stl10", "tinyimagenet", "imagenet12"]
    if args.dataset not in dataset_name_list:
        raise NotImplementedError
        
    '''Pre-fix moco version settings '''
    if args.model_version == "swav":
#         args.mlp = False
#         args.cos = False
#         args.K_dim = 128
        args.pairloader_option = "swav"
#         args.CLR_option = "multistep"


    # so that no need to set num_class
    if args.dataset == "cifar10":
        args.num_class = 10
        args.data_size = 32
    elif args.dataset == "svhn":
        args.num_class = 10
        args.data_size = 32
    elif args.dataset == "stl10":
        args.num_class = 10
        args.data_size = 96
    elif args.dataset == "tinyimagenet":
        args.num_class = 200
        args.data_size = 64
    elif args.dataset == "cifar100":
        args.num_class = 100
        args.data_size = 32
    elif args.dataset == "imagenet":
        args.num_class = 1000
        args.data_size = 224
    elif args.dataset == "imagenet12":
        args.num_class = 12
        args.data_size = 224
    else:
        raise("UNKNOWN DATASET!")