from load_cifar10_data import *
from load_cifar10 import *

CIFAR10_MEAN=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.]
CIFAR10_STD=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]
IMG_DIR = '../data'
CIFAR_IMAGES_NUM_TRAIN = 50000
TRAIN_BS = 256
NUM_WORKERS = 4
CROP_SIZE = 32

image_dir = "../data/"
batch_size = 256
data, targets = load_cifar10(batch_size, train=True, root=image_dir)
training_data_list, training_label_list, traindata_cls_counts = partition_data(data,
                                                                               targets,
                                                                               num_client=10,
                                                                               num_class=10, 
                                                                               partition = 'noniid', 
                                                                               beta=0.4)

pip_train = HybridTrainPipe_CIFAR(training_data_list[0],
                                  training_label_list[0],
                                  batch_size=TRAIN_BS, 
                                  num_threads=NUM_WORKERS, 
                                  device_id=0, 
                                  data_dir=IMG_DIR, 
                                  crop=CROP_SIZE, 
                                  world_size=1, 
                                  local_rank=0, 
                                  cutout=0)

train_loader = DALIDataloader(pipeline=pip_train, 
                              size=training_data_list[0].shape[0],
                              batch_size=TRAIN_BS,
                              onehot_label=True)

print("[DALI] train dataloader length: %d"%len(train_loader))
print('[DALI] start iterate train dataloader')

start = time.time()
for i, data in enumerate(train_loader):
    images = data['data'].cuda(non_blocking=True)
    labels = data['label'].cuda(non_blocking=True)
end = time.time()
train_time = end-start
print('[DALI] end train dataloader iteration')
print('[DALI] iteration time: %fs [train]' % (train_time))