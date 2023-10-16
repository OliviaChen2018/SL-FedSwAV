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

2023.7.28
1. 修改run_sflswav.py中测试阶段的代码, 增加dali方式下获取dataloader的步骤.
2. 修改functions/sflswav_functions.py文件中三个eval函数, 增加dali dataloader方式下数据的读取流程.
3. 修改models/resnet.py中模型的forward函数的bug.(之前改错了)
4. 新的问题: lr=0.006的时候loss突然就nan了,原因不明.之前都可以正常优化. 
5. 现在lr=0.0003可以正常优化, 并且一个epoch之后每个client的loss可以降到7.6.

2023.7.30
1. 修改loss的计算, 将所有client的表征拼接起来, 计算完所有表征的loss之后再进行反向传播并更新server端模型(之前的每个client计算loss之后都进行反向传播并更新server端模型的方式会受client数据异质性的影响, 阻碍模型收敛). 
2. 新增混合精度训练方式, 对训练过程进一步提速.
3. 对模型计算得到的表征进行归一化, 以消除表征向量长度对loss的影响.
4. 新的计算方式下需要更大的显存, 目前单卡只能一次性存放10个client的batch_size=2的20条数据, 训练效果不好. 下一步计划尝试DDP解决这个问题.

2023.8.3
1. 新增DDP方式下的dataloader sampler和读取方式.(将类CIFAR_INPUT_ITER的实例作为sampler传入DALIDataloader和pipeline, 并再在DALIDataloader类和所有Pipeline类中新增reset函数, 以便在训练过程中重置dataloader)
2. Dali/test_DDP.py为测试文件, 包含了在DaliDataloader方式下以DDP方式训练resnet模型完成cifar10分类任务的全过程. 
3. 下一步计划将DDP写入我的模型中.

2023.8.5
1. 将数据读取过程改为使用DDP-Dali的方式.

2023.8.7
1. Swav不work, loss不降.
    
2023.8.8
1. 修改functions/sflmoco_functions中MultiStepLR策略(非cos)下s_scheduler和c_scheduler_list的定义.
