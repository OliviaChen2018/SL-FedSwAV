修改记录

2023.7.10
将数据划分方式改为按Dirichlet分布划分。

2023.7.11
在run_sflmoco.py中修改gradient的返回方式。添加query_num变量，保存pool中每个client拥有的数据条数。在下发gradient时，根据每个client的数据条数进行梯度返回。
（222-227行  gradient_dict[j] = gradient[query_index[j-1]:query_index[j]]）

2023.7.16
新增multicropdataset.py文件，实现swav的multi-crop数据增强。
修改models/resnet.py中的模型结构，新增self.prototypes层，该层的参数作为prototypes，用于分配。
修改训练方式，将Moco对比loss改为计算SwAV的loss。(尚未完成)