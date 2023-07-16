'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

NUM_CHANNEL_GROUP = 4

class MobView(nn.Module): #池化，然后变成[batch_size, channel_size, 1]的向量
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        # out = F.avg_pool2d(input, 4)
        out = self.avgpool(input)
        batch_size = input.size(0)
        shape = (batch_size, -1)
        out = out.view(shape)
        return out

class WSConv2d(nn.Conv2d): # This module is taken from https://github.com/joe-siyuan-qiao/WeightStandardization

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def init_weights(m): # 初始化网络参数
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0) 
        #torch.nn.init的函数用来对神经网络的参数进行初始化
        if m.bias is not None: 
            m.bias.data.zero_() # 初始化bias，全部置0
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()

class BasicBlock(nn.Module):
    '''这是一个残差block'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, residual = True, WS = True):
        # 参数residual表示是否包含残差连接
        super(BasicBlock, self).__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if residual and (stride != 1 or in_planes != self.expansion*planes):
            # 如果需要网络包含残差连接，并且输入通道数(x的通道数)与输出通道数(out的通道数)不相同，或者步长不为1(步长不为1会使输入x的通道数≠out的通道数，导致无法直接相加)，则使用一个卷积层来改变通道的维数；否则直接返回输入(如76行所示)。
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_gn(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, residual = True, WS = True):
        super(BasicBlock_gn, self).__init__()
        self.residual = residual
        if WS:
            self.conv1 = WSConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.bn1 = nn.GroupNorm(num_groups = max(planes//NUM_CHANNEL_GROUP, 1), num_channels = planes)
        if WS:
            self.conv2 = WSConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups = max(planes//NUM_CHANNEL_GROUP, 1), num_channels = planes)

        self.shortcut = nn.Sequential()

        if residual and (stride != 1 or in_planes != self.expansion*planes):
            if WS:
                self.shortcut = nn.Sequential(
                    WSConv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(num_groups = max(self.expansion*planes//NUM_CHANNEL_GROUP, 1), num_channels = self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(num_groups = max(self.expansion*planes//NUM_CHANNEL_GROUP, 1), num_channels = self.expansion*planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out += self.shortcut(x)
        out = F.relu(out)
        return out


# Bottleneck是带有Bottleneck层的ResNet(1×1--3×3--1×1)
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, residual = True, WS = True):
        super(Bottleneck, self).__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if residual and (stride != 1 or in_planes != self.expansion*planes):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.residual:
            out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck_gn(nn.Module): #*_gn将BatchNorm换成了GroupNorm；将conv换成了WSConv
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, residual = True, WS = True):
        super(Bottleneck_gn, self).__init__()
        self.residual = residual
        if WS:
            self.conv1 = WSConv2d(in_planes, planes, kernel_size=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups = max(planes//NUM_CHANNEL_GROUP, 1), num_channels = planes)
        if WS:
            self.conv2 = WSConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups = max(planes//NUM_CHANNEL_GROUP, 1), num_channels = planes)
        if WS:
            self.conv3 = WSConv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        else:
            self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)


        self.bn3 = nn.GroupNorm(num_groups = max(self.expansion * planes//NUM_CHANNEL_GROUP, 1), num_channels = self.expansion * planes)

        self.shortcut = nn.Sequential()

        if residual and (stride != 1 or in_planes != self.expansion*planes):
            self.shortcut = nn.Sequential(
                WSConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups = max(self.expansion*planes//NUM_CHANNEL_GROUP, 1), num_channels = self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.residual:
            out += self.shortcut(x)
        out = F.relu(out)
        return out

# 普通的3x3卷积层
class conv3x3(nn.Module):
    def __init__(self, in_planes, planes, input_size=32):
        super(conv3x3, self).__init__()
        if input_size == 224:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=7, stride = 2, padding = 3, bias=False)
        elif input_size == 64:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride = 2, padding = 1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride = 1, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return out

# conv3x3_gn把卷积层Conv换成WSConv2d，BatchNorm2d换成了GroupNorm。
class conv3x3_gn(nn.Module):
    def __init__(self, in_planes, planes, input_size=32):
        super(conv3x3_gn, self).__init__()
        if input_size == 224:
            self.conv1 = WSConv2d(in_planes, planes, kernel_size=7, stride = 2, padding = 3, bias=False)
        elif input_size == 64:
            self.conv1 = WSConv2d(in_planes, planes, kernel_size=3, stride = 2, padding = 1, bias=False)
        else:
            self.conv1 = WSConv2d(in_planes, planes, kernel_size=3, stride = 1, padding = 1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups = max(planes//NUM_CHANNEL_GROUP, 1), num_channels = planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return out
    
class MultiPrototypes(nn.Module):
    # 创建多组不同的prototypes
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out

class ResNet(nn.Module):
    '''
    ResNet model 
    '''
    def __init__(self, feature, expansion = 1, num_client = 1, num_class = 10, nmb_prototypes = 0, input_size = 32):
        # feature==(local, cloud)。local包含client-side model的所有层；cloud包含server-side model的所有层，二者均为Sequential对象
        # num_client表示client的数量
        # num_class表示数据集中类别的数量
        # input_size表示数据集中每张图片的尺寸
        super(ResNet, self).__init__()
        self.current_client = 0
        self.num_client = num_client
        self.expansion = expansion
        self.local_list = []
        for i in range(num_client): # 初始化client-side models，保存到self.local_list中。
            if i == 0:
                self.local_list.append(feature[0]) # 用self.local_list存放client-side model
                self.local_list[0].apply(init_weights) # 对client-side model初始化
            else:  # 之后的每个初始client-side model都保存第一个的deepcopy
                new_copy = copy.deepcopy(self.local_list[0])
                self.local_list.append(new_copy)  

        self.cloud = feature[1] # 用self.cloud存放server-side model
        self.classifier = nn.Linear(512*expansion, num_class) # 分类层，用于evaluate。模型在表征层的输出通道数为512*expansion (即每张图片有512*expansion个特征)。(这里只是为了定义classifier层，所以这么写，实际上这一层在training和eval阶段各有定义)
#         self.prototypes = nn.Linear(512*expansion, num_class) # 分类层，用于evaluate。模型在表征层的输出通道数为512*expansion (即每张图片有512*expansion个特征)。
        self.cloud_classifier_merge = False  #cloud_classifier_merge标志为True，表示当前的server-side model已经合并了分类层
        self.original_num_cloud = self.get_num_of_cloud_layer()

        # Initialize weights
        self.cloud.apply(init_weights) # 初始化server-side model的参数
        self.classifier.apply(init_weights) # 初始化分类层的参数
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1)) # 定义一个平均池化层——自适应均值池化。每个通道的输出变为1×1
        # prototype layer  (定义prototypes层)
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(num_class, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(num_class, nmb_prototypes, bias=False)
        self.prototypes.apply(init_weights) # 初始化分类层的参数
        
#     # MocoSFL的
#     def forward(self, x, client_id = 0):
#         if self.cloud_classifier_merge:
#             x = self.local_list[client_id](x) #x为第client_id个client的数据，因此使用第client_id个client-side model计算表征。
#             x = self.cloud(x) #server-side model只能接触到x的表征，而不接触数据本身。
#         else:
#             x = self.local_list[client_id](x)
#             x = self.cloud(x)
#             # x = F.avg_pool2d(x, 4)
#             x = self.avg_pool(x)
#             x = x.view(x.size(0), -1)
#             x = self.classifier(x)
#         return x
    
    # FedSwAV的
    def forward(self, x, client_id = 0):
        if self.cloud_classifier_merge:
            if not isinstance(x, list):
                x = [x]
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 0)
            start_idx = 0
            for end_idx in idx_crops:
                _out = self.local_list[client_id](torch.cat(x[start_idx: end_idx]).cuda(non_blocking=True)) 
                if start_idx == 0:
                    output = _out
                else:
                    output = torch.cat((output, _out)) 
                start_idx = end_idx
            
            x = self.cloud(output) #server-side model只能接触到x的表征，而不接触数据本身。
            
            x = nn.functional.normalize(x, dim=1, p=2)

            if self.prototypes is not None:  # prototypes将每张图片的特征数量变为nmb_prototypes
                return x, self.prototypes(x)   #self.prototypes(x)：[B, nmb_prototypes]
        else:
            x = self.local_list[client_id](x)
            x = self.cloud(x)
            # x = F.avg_pool2d(x, 4)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x
    
    
    def __call__(self, x, client_id = 0):
        return self.forward(x, client_id)

    def merge_classifier_cloud(self): # 给server-side model加上分类层，并设置标志为True
        self.cloud_classifier_merge = True
        cloud_list = list(self.cloud.children()) #cloud_list为self.cloud中所有层组成的list
        cloud_list.append(MobView())  #加一个MobView层(该层的操作是池化，池化后变成[batch_size, channel_size, 1]的矩阵)
        cloud_list.append(self.classifier) # 加上分类层
        self.cloud = nn.Sequential(*cloud_list) # 重新将新的cloud_list变为Sequential

    def unmerge_classifier_cloud(self): # 给server-side model去掉分类层，并设置标志为False
        self.cloud_classifier_merge = False
        cloud_list = list(self.cloud.children())
        orig_cloud_list = []
        for i, module in enumerate(cloud_list):
            if "MobView" in str(module): #MobView及之后的层不要了
                break
            else:
                orig_cloud_list.append(module)
        self.cloud = nn.Sequential(*orig_cloud_list)

    def get_num_of_cloud_layer(self):
        num_of_cloud_layer = 0
        if not self.cloud_classifier_merge:
            list_of_layers = list(self.cloud.children())
            for i, module in enumerate(list_of_layers):
                if "conv3x3" in str(module) or "Linear" in str(module) or "BasicBlock" in str(module) or "BottleNeck" in str(module):
                    num_of_cloud_layer += 1
            num_of_cloud_layer += 1  # 为什么没有分类层的时候要多加1？
        else:
            list_of_layers = list(self.cloud.children())
            for i, module in enumerate(list_of_layers):
                if "conv3x3" in str(module) or "Linear" in str(module) or "BasicBlock" in str(module) or "BottleNeck" in str(module):
                    num_of_cloud_layer += 1
        return num_of_cloud_layer

    def recover(self): # recover和resplit这两个函数都没有用。
        if self.cloud_classifier_merge:
            self.resplit(self.original_num_cloud)
            self.unmerge_classifier_cloud()
            

    def resplit(self, num_of_cloud_layer):
        if not self.cloud_classifier_merge:
            self.merge_classifier_cloud()
            
        for i in range(self.num_client):
            list_of_layers = list(self.local_list[i].children())
            list_of_layers.extend(list(self.cloud.children()))
            entire_model_list = copy.deepcopy(list_of_layers)
            total_layer = 0
            for _, module in enumerate(entire_model_list):
                if "conv3x3" in str(module) or "Linear" in str(module) or "BasicBlock" in str(module) or "BottleNeck" in str(module):
                    total_layer += 1
            
            num_of_local_layer = (total_layer - num_of_cloud_layer)
            local_list = []
            local_count = 0
            cloud_list = []
            for _, module in enumerate(entire_model_list):
                if "conv3x3" in str(module) or "Linear" in str(module) or "BasicBlock" in str(module) or "BottleNeck" in str(module):
                    local_count += 1
                if local_count <= num_of_local_layer:
                    local_list.append(module)
                else:
                    cloud_list.append(module)
            self.local_list[i] = nn.Sequential(*local_list)
        self.cloud = nn.Sequential(*cloud_list)

    def get_smashed_data_size(self, batch_size = 1, input_size = 32):
        # 怎么得到client-side model的输出size？向client-side model输入一个随机矩阵，取输出的size。
        self.local_list[0].eval()
        with torch.no_grad():
            noise_input = torch.randn([batch_size, 3, input_size, input_size])
            try:
                device = next(self.local_list[0].parameters()).device
                noise_input = noise_input.to(device)
            except:
                pass
            smashed_data = self.local_list[0](noise_input)
        return smashed_data.size()
    
    

# 构造client-side model和server-side model的层
def make_layers(block, layer_list, cutting_layer, adds_bottleneck = False, bottleneck_option = "C8S1", group_norm = False, input_size = 32, residual = True, WS = True):
    # layer_list==[2, 2, 2, 2]，以列表的形式传入每一层内部包含的resnet block的数量
    # block: resnet18/34用的BasicBlock，其余用的Bottleneck
    layers = []
    current_image_dim = input_size # input_size是一个由数据集决定的已知变量，表示图像的size
    count = 1
    if not group_norm: # 第一层是不加残差连接的普通卷积层
        layers.append(conv3x3(3, 64, input_size))
    else:  # conv3x3_gn把卷积层换成WSConv2d，BatchNorm2d换成了GroupNorm。
        layers.append(conv3x3_gn(3, 64, input_size))
    in_planes = 64

    strides = [1] + [1]*(layer_list[0]-1) # strides==[1,1]
    for stride in strides:
        if count >= cutting_layer: # server-side model的层包含残差连接
            residual = True
        layers.append(block(in_planes, 64, stride, residual, WS))
        count += 1
        current_image_dim = current_image_dim // stride  
        # 这个残差模块的图像输出尺寸为原始的1/stride

        in_planes = 64 * block.expansion 
        # 下一个block的输入通道维度变为out_planes*block.expansion

    strides = [2] + [1]*(layer_list[1]-1)
    for stride in strides:
        if count >= cutting_layer:
            residual = True
        layers.append(block(in_planes, 128, stride, residual, WS))
        count += 1
        current_image_dim = current_image_dim // stride
        in_planes = 128 * block.expansion

    strides = [2] + [1]*(layer_list[2]-1)
    for stride in strides:
        if count >= cutting_layer:
            residual = True
        layers.append(block(in_planes, 256, stride, residual, WS))
        count += 1
        current_image_dim = current_image_dim // stride
        in_planes = 256 * block.expansion

    strides = [2] + [1]*(layer_list[3]-1)
    for stride in strides:
        if count >= cutting_layer:
            residual = True
        layers.append(block(in_planes, 512, stride, residual, WS))
        count += 1
        current_image_dim = current_image_dim // stride
        in_planes = 512 * block.expansion # 最终模型输出的通道数为512 * block.expansion，即每张图片有512 * block.expansion个特征。
    # 以上是构建整个resnet网络。下面开始分割和构建client-side和server-side model
    try: 
        local_layer_list = layers[:cutting_layer]  # cutting_layer以下是client-side model的所有层；
        cloud_layer_list = layers[cutting_layer:] # cutting_layer以上是server-side model的所有层
    except:
        print("Cutting layer is greater than overall length of the ResNet arch! set cloud to empty list")
        local_layer_list = layers[:]
        cloud_layer_list = []

    # Adding a pair of bottleneck layers for communication-efficiency (不加也可以)
    temp_local = nn.Sequential(*local_layer_list)
    with torch.no_grad():
        noise_input = torch.randn([1, 3, input_size, input_size]) 
        #第一个参数表示共有1条数据；因为数据集是彩色图像,所以第二个参数是3；
        smashed_data = temp_local(noise_input)
    input_nc = smashed_data.size(1) # (input_nc==in_planes)

    local = []
    cloud = []
    # adds_bottleneck值取决于bottleneck_option（指定了则adds_bottleneck=True, 否则为False）
    if adds_bottleneck: # to enable gooseneck, simply copy below to other architecture
        print("original channel size of smashed-data is {}".format(input_nc))
        try: # 按照bottleneck_option参数设置bottleneck卷积核的kernal_size和stride
            if "noRELU" in bottleneck_option or "norelu" in bottleneck_option or "noReLU" in bottleneck_option:
                relu_option = False
            else:
                relu_option = True
            if "K" in bottleneck_option:
                bn_kernel_size = int(bottleneck_option.split("C")[0].split("K")[1])
            else:
                bn_kernel_size = 3
            bottleneck_channel_size = int(bottleneck_option.split("S")[0].split("C")[1])
            if "S" in bottleneck_option:
                bottleneck_stride = int(bottleneck_option.split("S")[1])
            else:
                bottleneck_stride = 1
        except:
            print("auto extract bottleneck option fail (format: CxSy, x = [1, max_channel], y = {1, 2}), set channel size to 8 and stride to 1")
            bn_kernel_size = 3
            bottleneck_channel_size = 8
            bottleneck_stride = 1
            relu_option = True
        # cleint-side bottleneck
        if bottleneck_stride == 1:
            local += [nn.Conv2d(input_nc, bottleneck_channel_size, kernel_size=bn_kernel_size, padding=bn_kernel_size//2, stride= 1)]
        elif bottleneck_stride >= 2:
            local += [nn.Conv2d(input_nc, bottleneck_channel_size, kernel_size=3, padding=1, stride= 2)]
            for _ in range(int(np.log2(bottleneck_stride//2))):
                if relu_option:
                    local += [nn.ReLU()]
                local += [nn.Conv2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, padding=1, stride= 2)]
        if relu_option:
            local += [nn.ReLU()]
        # server-side bottleneck
        if bottleneck_stride == 1:
            cloud += [nn.Conv2d(bottleneck_channel_size, input_nc, kernel_size=bn_kernel_size, padding=bn_kernel_size//2, stride= 1)]
        elif bottleneck_stride >= 2:
            for _ in range(int(np.log2(bottleneck_stride//2))):
                cloud += [nn.ConvTranspose2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, output_padding=1, padding=1, stride= 2)]
                if relu_option:
                    cloud += [nn.ReLU()]
            cloud += [nn.ConvTranspose2d(bottleneck_channel_size, input_nc, kernel_size=3, output_padding=1, padding=1, stride= 2)]
        if relu_option:
            cloud += [nn.ReLU()]
        print("added bottleneck, new channel size of smashed-data is {}".format(bottleneck_channel_size))
        input_nc = bottleneck_channel_size
    local_layer_list += local  # 在client-side model后面加几层
    cloud_layer_list = cloud + cloud_layer_list # 在server-side model后面加几层
    local = nn.Sequential(*local_layer_list)
    cloud = nn.Sequential(*cloud_layer_list)

    # local表示client-side model的所有层的Sequential
    # cloud表示server-side model的所有层的Sequential
    return local, cloud  

def ResNet18(cutting_layer, num_client = 1, num_class = 10, nmb_prototypes = 0, adds_bottleneck = False, bottleneck_option = "C8S1", batch_norm=True, group_norm = False, input_size = 32, c_residual = True, WS = True):
    if not group_norm:
        return ResNet(make_layers(BasicBlock, 
                                  [2, 2, 2, 2], 
                                  cutting_layer, 
                                  adds_bottleneck = adds_bottleneck, 
                                  bottleneck_option = bottleneck_option, 
                                  input_size = input_size, 
                                  residual = c_residual, 
                                  WS = WS), 
                      num_client = num_client, 
                      num_class = num_class, 
                      nmb_prototypes = nmb_prototypes,
                      input_size = input_size)
    # [2, 2, 2, 2]是ResNet18指定的，ResNet34/50是[3,4,6,3]
    else:
        return ResNet(make_layers(BasicBlock_gn, 
                                  [2, 2, 2, 2], 
                                  cutting_layer, 
                                  adds_bottleneck = adds_bottleneck, 
                                  bottleneck_option = bottleneck_option, 
                                  group_norm = group_norm, 
                                  input_size = input_size, 
                                  residual = c_residual, 
                                  WS = WS), 
                      num_client = num_client, 
                      num_class = num_class, 
                      nmb_prototypes = nmb_prototypes,
                      input_size = input_size)

def ResNet34(cutting_layer, num_client = 1, num_class = 10, adds_bottleneck = False, bottleneck_option = "C8S1", batch_norm=True, group_norm = False, input_size = 32, c_residual = True, WS = True):
    if not group_norm:
        return ResNet(make_layers(BasicBlock, [3, 4, 6, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, input_size = input_size, residual = c_residual, WS = WS), num_client = num_client, num_class = num_class, input_size = input_size)
    else:
        return ResNet(make_layers(BasicBlock_gn, [3, 4, 6, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, group_norm = group_norm, input_size = input_size, residual = c_residual, WS = WS), num_client = num_client, num_class = num_class, input_size = input_size)

def ResNet50(cutting_layer, num_client = 1, num_class = 10, adds_bottleneck = False, bottleneck_option = "C8S1", batch_norm=True, group_norm = False, input_size = 32, c_residual = True, WS = True):
    if not group_norm:
        return ResNet(make_layers(Bottleneck, [3, 4, 6, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, input_size = input_size, residual = c_residual, WS = WS), expansion= 4, num_client = num_client, num_class = num_class, input_size = input_size)
    else:
        return ResNet(make_layers(Bottleneck_gn, [3, 4, 6, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, group_norm = group_norm, input_size = input_size, residual = c_residual, WS = WS), expansion= 4, num_client = num_client, num_class = num_class, input_size = input_size)

def ResNet101(cutting_layer, num_client = 1, num_class = 10, adds_bottleneck = False, bottleneck_option = "C8S1", batch_norm=True, group_norm = False, input_size = 32, c_residual = True, WS = True):
    if not group_norm:
        return ResNet(make_layers(Bottleneck, [3, 4, 23, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, input_size = input_size, residual = c_residual, WS = WS), expansion= 4, num_client = num_client, num_class = num_class, input_size = input_size)
    else:
        return ResNet(make_layers(Bottleneck_gn, [3, 4, 23, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, group_norm = group_norm, input_size = input_size, residual = c_residual, WS = WS), expansion= 4, num_client = num_client, num_class = num_class, input_size = input_size)

def ResNet152(cutting_layer, num_client = 1, num_class = 10, adds_bottleneck = False, bottleneck_option = "C8S1", batch_norm=True, group_norm = False, input_size = 32, c_residual = True, WS = True):
    if not group_norm:
        return ResNet(make_layers(Bottleneck, [3, 8, 36, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, input_size = input_size, residual = c_residual, WS = WS), expansion= 4, num_client = num_client, num_class = num_class, input_size = input_size)
    else:
        return ResNet(make_layers(Bottleneck_gn, [3, 8, 36, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option, group_norm = group_norm, input_size = input_size, residual = c_residual, WS = WS), expansion= 4, num_client = num_client, num_class = num_class, input_size = input_size)

