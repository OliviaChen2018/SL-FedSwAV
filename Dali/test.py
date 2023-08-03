from load_cifar10_data import *
from cifar_Dali_Dataset import *

CIFAR10_MEAN=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.]
CIFAR10_STD=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]
IMG_DIR = '../data'
CIFAR_IMAGES_NUM_TRAIN = 50000
TRAIN_BS = 256 
NUM_WORKERS = 4
CROP_SIZE = 32

image_dir = "../data/"
batch_size = 256
NMB_CROPS = [2,6]
SIZE_CROPS = [224,96]
MIN_SCALE = [0.14,0.05]
MAX_SCALE = [1,0.14]
data, targets = load_cifar10(batch_size, train=True, root=image_dir)
training_data_list, training_label_list, traindata_cls_counts = partition_data(data,
                                                                               targets,
                                                                               num_client=10,
                                                                               num_class=10, 
                                                                               partition = 'noniid', 
                                                                               beta=0.4)

pip_train = DaliTrainPipe_CIFAR_multicrop(training_data_list[0],
                                  training_label_list[0],
                                  batch_size=TRAIN_BS, 
                                  num_threads=NUM_WORKERS, 
                                  device_id=0, 
                                  nmb_crops = NMB_CROPS,
                                  size_crops=SIZE_CROPS, 
                                  min_scale_crops=MIN_SCALE,
                                  max_scale_crops=MAX_SCALE,  
                                  world_size=1, 
                                  local_rank=0, 
                                  cutout=0)

train_loader = DALIDataloader(pipeline=pip_train, 
                              size=training_data_list[0].shape[0],
                              batch_size=TRAIN_BS)

print("[DALI] train dataloader length: %d"%len(train_loader))
print('[DALI] start iterate train dataloader')

start = time.time()
for i, data in enumerate(train_loader):
    images = data['data'].cuda(non_blocking=True) # images.size()==torch.tensor([bs, 3, 1, all])
    labels = data['label'].cuda(non_blocking=True)
#     pdb.set_trace()
    images_dict = {i: None for i in range(sum(NMB_CROPS))}
    ptr = 0
    column_add_pre = 0
    for crop_id in range(len(NMB_CROPS)): # 0,1
        for num_id in range(NMB_CROPS[crop_id]): #NMB_CROPS[0]==2
            column_add = column_add_pre+SIZE_CROPS[crop_id] * SIZE_CROPS[crop_id]
            images_dict[ptr] = images[:,:, :, column_add_pre: column_add]
            images_dict[ptr] = images_dict[ptr].reshape([batch_size, 3, SIZE_CROPS[crop_id], -1])
            column_add_pre = column_add
#             print(f"column_add_pre=:{column_add_pre}")
#             print(f"column_add=:{column_add}")
            print(f"size of images_dict[{ptr}]: {images_dict[ptr].size()}")
            ptr+=1
#     pdb.set_trace()       
#     print(f"size of images: {images.size()}")

end = time.time()
train_time = end-start
print('[DALI] end train dataloader iteration')
print('[DALI] iteration time: %fs [train]' % (train_time))