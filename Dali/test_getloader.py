from load_cifar10_data import *
from cifar_Dali_Dataset import *
from get_dali_dataloader import get_cifar10_dali

CIFAR10_MEAN=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.]
CIFAR10_STD=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]
IMG_DIR = '../data'
CIFAR_IMAGES_NUM_TRAIN = 50000
TRAIN_BS = 256
NUM_WORKERS = 4
CROP_SIZE = 32

batch_size = 256
NMB_CROPS = [2,6]
SIZE_CROPS = [224,96]
MIN_SCALE = [0.14,0.05]
MAX_SCALE = [1,0.14]
PAIR = "swav"


if PAIR != "None":
    train_loader, traindata_cls_counts, mem_loader, test_loader = get_cifar10_dali(size_crops=SIZE_CROPS, nmb_crops=NMB_CROPS, min_scale_crops=MIN_SCALE, max_scale_crops=MAX_SCALE, batch_size=batch_size, num_workers=4, num_client = 10, data_proportion = 1.0, noniid_ratio =1.0, pairloader_option = PAIR, partition = 'noniid', partition_beta = 0.4, hetero = False, path_to_data = IMG_DIR)

print("[DALI] train dataloader length: %d"%len(train_loader))
print('[DALI] start iterate train dataloader')

start = time.time()
for client_id in range(len(train_loader)):
    print(f"第{client_id}个client的dataloader的长度为:{len(train_loader[client_id])}")
    for i, data in enumerate(train_loader[client_id]):
    #     images = data['data'].cuda(non_blocking=True) # images.size()==torch.tensor([bs, 3, 1, all])
    #     labels = data['label'].cuda(non_blocking=True)
        images = data['data']# images.size()==torch.tensor([bs, 3, 1, all])
        labels = data['label']
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


if PAIR != "None":
    print("[DALI] mem dataloader length: %d"%len(mem_loader))
    print('[DALI] start iterate mem dataloader')
    start = time.time()
    for i, data in enumerate(mem_loader):
#         images = data['data'].cuda(non_blocking=True) 
#         labels = data['label'].cuda(non_blocking=True)
        images = data['data']
        labels = data['label']
    #     pdb.set_trace()       
    #     print(f"size of images: {images.size()}")
    end = time.time()
    test_time = end-start
    print('[DALI] end mem dataloader iteration')
    print('[DALI] iteration time: %fs [mem]' % (mem_time))


print("[DALI] test dataloader length: %d"%len(test_loader))
print('[DALI] start iterate test dataloader')
start = time.time()
for i, data in enumerate(test_loader):
    images = data['data'] # images.size()==torch.tensor([bs, 3, 1, all])
    labels = data['label']
#     pdb.set_trace()       
#     print(f"size of images: {images.size()}")
end = time.time()
test_time = end-start
print('[DALI] end test dataloader iteration')
print('[DALI] iteration time: %fs [test]' % (test_time))