from cifar10 import *
import pdb

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
CIFAR_IMAGES_NUM_TRAIN = 50000
CIFAR_IMAGES_NUM_TEST = 10000
IMG_DIR = '../data'
TRAIN_BS = 256
TEST_BS = 200
NUM_WORKERS = 4
CROP_SIZE = 32

# if __name__ == '__main__':
# iteration of DALI dataloader
pip_train = HybridTrainPipe_CIFAR(batch_size=TRAIN_BS, 
                                  num_threads=NUM_WORKERS, 
                                  device_id=0, 
                                  data_dir=IMG_DIR, 
                                  crop=CROP_SIZE, 
                                  world_size=1, 
                                  local_rank=0, 
                                  cutout=0)
pip_test = HybridTestPipe_CIFAR(batch_size=TEST_BS, 
                                num_threads=NUM_WORKERS, 
                                device_id=0, 
                                data_dir=IMG_DIR, 
                                crop=CROP_SIZE, 
                                size=CROP_SIZE, 
                                world_size=1, 
                                local_rank=0)
train_loader = DALIDataloader(pipeline=pip_train, 
                              size=CIFAR_IMAGES_NUM_TRAIN, 
                              batch_size=TRAIN_BS, 
                              onehot_label=True)
test_loader = DALIDataloader(pipeline=pip_test, 
                             size=CIFAR_IMAGES_NUM_TEST,
                             batch_size=TEST_BS, 
                             onehot_label=True)
# 也可以直接使用DALIGenericIterator类
# 即train_loader = DALIGenericIterator(pipeline=pip_train, output_map=["data", "label"])
print("[DALI] train dataloader length: %d"%len(train_loader))
print('[DALI] start iterate train dataloader')

start = time.time()
for i, data in enumerate(train_loader):
    images = data['data'].cuda(non_blocking=True)
    labels = data['label'].cuda(non_blocking=True)
end = time.time()
train_time = end-start
print('[DALI] end train dataloader iteration')

print("[DALI] test dataloader length: %d"%len(test_loader))
print('[DALI] start iterate test dataloader')
start = time.time()
for i, data in enumerate(test_loader):
    images = data['data'].cuda(non_blocking=True)
    labels = data['label'].cuda(non_blocking=True)
#     images = data[0].cuda(non_blocking=True)
#     labels = data[1].cuda(non_blocking=True)
end = time.time()
test_time = end-start
print('[DALI] end test dataloader iteration')
print('[DALI] iteration time: %fs [train],  %fs [test]' % (train_time, test_time))


# iteration of PyTorch dataloader
transform_train = transforms.Compose([
    transforms.RandomCrop(CROP_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
train_dst = CIFAR10(root=IMG_DIR, train=True, download=True, transform=transform_train)
# print(f"size of data:{train_dst.data}")
# print(f"size of labels: {len(train_dst.targets)}")
train_loader = torch.utils.data.DataLoader(train_dst, batch_size=TRAIN_BS, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
test_dst = CIFAR10(root=IMG_DIR, train=False, download=True, transform=transform_test)
test_iter = torch.utils.data.DataLoader(test_dst, batch_size=TEST_BS, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)
print("[PyTorch] train dataloader length: %d"%len(train_loader))
print('[PyTorch] start iterate train dataloader')
start = time.time()
for i, data in enumerate(train_loader):
    images = data[0].cuda(non_blocking=True)
    labels = data[1].cuda(non_blocking=True)
end = time.time()
train_time = end-start
print('[PyTorch] end train dataloader iteration')

print("[PyTorch] test dataloader length: %d"%len(test_iter))
print('[PyTorch] start iterate test dataloader')
start = time.time()
for i, data in enumerate(test_iter):
    images = data[0].cuda(non_blocking=True)
    labels = data[1].cuda(non_blocking=True)
end = time.time()
test_time = end-start
print('[PyTorch] end test dataloader iteration')
print('[PyTorch] iteration time: %fs [train],  %fs [test]' % (train_time, test_time))