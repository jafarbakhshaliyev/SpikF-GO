#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate tsf

NVIDIA_LIBS=$(find /home/bakhshaliyev/anaconda3/envs/semm-env/lib/python3.10/site-packages/nvidia -name "lib" -type d | tr '\n' ':')
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:$LD_LIBRARY_PATH

cd /home/bakhshaliyev/STSF/code

echo "GPU Status:"
nvidia-smi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi



python maints.py \
    --model FGN \
    --data traffic \
    --feature_size  963  \
    --embed_size 128 \
    --proj_dim 32 \
    --hidden_size 256 \
    --batch_size 2 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --patch_num 4 \
    --patch_dim 16 \
    --T 16 \
    --tau 2.0 \
    --alpha 1.0 \
    --device cuda:0 >logs/LongForecasting/Traffic_12_FGN.log


python maints.py \
    --model SpikF \
    --data traffic \
    --feature_size  963  \
    --embed_size 128 \
    --proj_dim 32 \
    --hidden_size 256 \
    --batch_size 2 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --patch_num 4 \
    --patch_dim 16 \
    --blocks 2\
    --T 16 \
    --tau 2.0 \
    --alpha 1.0 \
    --device cuda:0 >logs/LongForecasting/Traffic_12_SpikF.log


python maints.py \
    --model iSpikformer \
    --data traffic \
    --feature_size  963  \
    --embed_size 128 \
    --proj_dim 32 \
    --hidden_size 256 \
    --batch_size 2 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --patch_num 4 \
    --patch_dim 16 \
    --T 4 \
    --tau 2.0 \
    --alpha 1.0 \
    --device cuda:0 >logs/LongForecasting/Traffic_12_iSpikformer.log


python maints.py \
    --model TSGRU \
    --data traffic \
    --feature_size  963  \
    --embed_size 128 \
    --proj_dim 32 \
    --hidden_size 128 \
    --batch_size 2 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --patch_num 4 \
    --patch_dim 16 \
    --T 4 \
    --tau 2.0 \
    --alpha 1.0 \
    --blocks 1 \
    --device cuda:0 >logs/LongForecasting/Traffic_12_TSGRU.log



python maints.py \
    --model TSTCN \
    --data traffic \
    --feature_size  963  \
    --embed_size 128 \
    --proj_dim 32 \
    --hidden_size 64 \
    --batch_size 2 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --patch_num 4 \
    --patch_dim 16 \
    --T 4 \
    --tau 2.0 \
    --alpha 1.0 \
    --blocks 3 \
    --kernel_size 32 \
    --device cuda:0 >logs/LongForecasting/Traffic_12_TSTCN.log


python maints.py \
    --model TSFormer \
    --data traffic \
    --feature_size  963  \
    --embed_size 128 \
    --proj_dim 32 \
    --hidden_size 128 \
    --batch_size 2 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --patch_num 4 \
    --patch_dim 16 \
    --T 4 \
    --tau 2.0 \
    --alpha 1.0 \
    --device cuda:0 >logs/LongForecasting/Traffic_12_TSFormer.log
