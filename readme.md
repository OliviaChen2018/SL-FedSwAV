修改记录

2023.7.10
1. 将数据划分方式改为按Dirichlet分布划分。

2023.7.11
1. 在run_sflmoco.py中修改gradient的返回方式。添加query_num变量，保存pool中每个client拥有的数据条数。在下发gradient时，根据每个client的数据条数进行梯度返回。（222-227行  gradient_dict[j] = gradient[query_index[j-1]:query_index[j]]）

2023.7.15
1. 新增multicropdataset.py文件，实现swav的multi-crop数据增强；
2. 修改models/resnet.py中的模型结构，新增self.prototypes层，该层的参数作为prototypes，用于分配。

2023.7.16
1. 修改训练方式，将Moco对比loss改为计算SwAV的loss。(尚未完成)

2023.7.19
1. 修改了create_sflmococlient_instance类的forward的实现,以便读取multi-crop数据.
2. 完成了对swav loss计算的修改.
3. 模型开始顺利运行了,但目前训练速度非常慢,打算尝试nvidia.dali对数据增强步骤进行加速.

2023.7.21
1. 上传了别人写的dali模板.
2. dali_load_data.ipynb实现了:1)对cafar10数据集的读取; 2)在读取到的原始数据上根据Dirichlet分布进行划分.

2023.7.23
1. 完成了在cifar10原始数据上直接做non-IID数据划分,以及使用dali库实现multicrop数据增强的整个流程;
2. Dali/multicrop_from_dali_baseline文件夹中实现了(load_cifar10_data.py定义了读取cifar10原始数据集的方法load_cifar10(),以及对读取的数据按Dirichlet分布进行划分的方法partition(); Dali_Dataloader.py定义Dataloader类; cifar10_Dali_Dataset.py定义数据集的pipeline, 实现数据增强,并返回整合后的增强样本 )
3. Dali/baseline_byGithub文件夹中的文件为Github(https://github.com/tanglang96/DataLoaders_DALI/tree/master )上的cifar10数据集的读取baseline (修改了其中的bug).

2023.7.25
1. 新增Dali/get_dali_dataloader.py文件, 使得整个模型的数据增强过程都使用nvidia.dali实现.(最终返回3个dataloader,与torch.utils.Dataloader方式下的返回值一致)
2. 在run_sflswav.py文件中,新增dali方式下读取数据后的维度处理操作(if args.use_dali部分), 使最终得到的images与torch.utils.Dataloader方式下的结果保持一致.
3. 下一步打算修改swav方式的核心计算部分, 不再分开对每个client的loss执行反向传播计算梯度,而是将所有client的embedding拼接在一起, 计算多个loss, 但只保存一张梯度计算图, 只执行总loss的反向传播.

2023.7.26
1. 修改sflswav_functions.py中compute_swav函数对loss的计算方式.

2023.7.27
1. 修改sflswav_functions.py中compute_swav函数对loss的计算方式, 以及run_sflswav.py中server端模型s_optimizer的更新策略(由之前的所有client的loss计算完成之后再更新,修改为每个client都对server端模型进行更新)
2. 修改后的更新方式允许server端和client端模型的学习率保持一致,并且使用一个较大的值. (之前的方式: 每个client的loss计算完成后执行反向传播,但不更新模型. 等所有client得到的梯度汇总之后再一次性更新server端模型, 相当于模拟大batch_size的训练方式. 这种方式导致server端的模型使用lr==0.06这样较大的值更新时,loss会变成nan, 并且最终效果很差.)