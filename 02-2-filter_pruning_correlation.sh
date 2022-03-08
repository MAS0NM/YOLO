model=resnet20_B
###
 # Copyright (c) 2021 Leinao
 # All rights reserved.
 # 
 # @Author: username
 # @Date: 2022-02-21 15:30:26
 # @LastEditors: username
 # @LastEditTime: 2022-02-21 15:31:11
### 
fp_flops=0.1
fp_params=0
target_fp_flops=0.5
target_fp_params=0.0
flops=1
params=1
avg_bit_weights=32
avg_bit_fm=32
max_bw=8

CUDA_VISIBLE_DEVICES=3 python 02-2-filter_pruning_correlation.py \
--arch $model \
--data_set cifar10 \
--data_path ./EMO-NN-master/dataset/cifar10 \
--job_dir ./EMO-NN-master/experiment/cifar10/$model/02-filter_pruning/correlation \
--baseline True \
--baseline_model ./EMO-NN-master/experiment/cifar10/$model/baseline/checkpoint/model_best.pt  \
-m_f_fp $fp_flops -m_p_fp $fp_params -t_f_fp $target_fp_flops -t_p_fp $target_fp_params  \
--optimizer SGD --lr 0.01 --weight_decay 1e-5  \
--label-smoothing 0.0 \
--num_epochs 30 \
