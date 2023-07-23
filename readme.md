修改记录

2023.7.10
将数据划分方式改为按Dirichlet分布划分。

2023.7.11
在run_sflmoco.py中修改gradient的返回方式。添加query_num变量，保存pool中每个client拥有的数据条数。在下发gradient时，根据每个client的数据条数进行梯度返回。
（222-227行  gradient_dict[j] = gradient[query_index[j-1]:query_index[j]]）

2023.7.15
新增multicropdataset.py文件，实现swav的multi-crop数据增强；
修改models/resnet.py中的模型结构，新增self.prototypes层，该层的参数作为prototypes，用于分配。

2023.7.16
修改训练方式，将Moco对比loss改为计算SwAV的loss。(尚未完成)

2023.7.19
1.修改了create_sflmococlient_instance类的forward的实现,以便读取multi-crop数据.
2.完成了对swav loss计算的修改.
3.模型开始顺利运行了,但目前训练速度非常慢,打算尝试nvidia.dali对数据增强步骤进行加速.

2023.7.21
1.上传了别人写的dali模板.
2.dali_load_data.ipynb实现了:1)对cafar10数据集的读取; 2)在读取到的原始数据上根据Dirichlet分布进行划分.

2023.7.23
1. 完成了使用dali库对cifar10做non-IID数据划分,以及实现multicrop数据增强的整个流程;
2. Dali/multicrop_from_dali_baseline文件夹中实现了(load_cifar10_data.py定义了读取cifar10原始数据集的方法load_cifar10(),以及对读取的数据按Dirichlet分布进行划分的方法partition(); Dali_Dataloader.py定义Dataloader类; cifar10_Dali_Dataset.py定义数据集的pipeline, 实现数据增强,并返回整合后的增强样本 )
3. Dali/baseline_byGithub文件夹中的文件为Github(https://github.com/tanglang96/DataLoaders_DALI/tree/master )上的cifar10数据集的读取baseline (修改了其中的bug).