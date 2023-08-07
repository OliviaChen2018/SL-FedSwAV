#!/bin/bash
cd "$(dirname "$0")"
cd ../../

#fixed arguments
num_epoch=200
lr=0.06
moco_version=V2
arch=ResNet18
non_iid_list="1.0"
cutlayer_list="1"
num_client_list="10"
dataset=cifar10
loss_threshold=0.0
ressfl_alpha=0.0
bottleneck_option=None
batch_size=16
avg_freq=10
device='cuda:6'
pairloader_option='mocov2'
for num_client in $num_client_list; do
        for noniid_ratio in $non_iid_list; do
                for cutlayer in $cutlayer_list; do
                        output_dir="./outputs/simcosfl${moco_version}_${arch}_${dataset}_cut${cutlayer}_bnl${bottleneck_option}_client${num_client}_nonIID${noniid_ratio}_dirichlet"
                         torchrun --nproc_per_node=4 run_sflmoco.py\
                                --num_client ${num_client} --lr ${lr} --cutlayer ${cutlayer} --num_epoch ${num_epoch}\
                                --noniid_ratio ${noniid_ratio}  --hetero --output_dir ${output_dir}\
                                --moco_version ${moco_version} --arch ${arch} --dataset ${dataset} --loss_threshold ${loss_threshold} --pairloader_option ${pairloader_option}\
                                --ressfl_alpha ${ressfl_alpha} --bottleneck_option ${bottleneck_option} --batch_size ${batch_size}\
                                --avg_freq ${avg_freq}\
                                --is_distributed --cos
                done
        done
done
## for test, add --resume --attack --device ${device- 