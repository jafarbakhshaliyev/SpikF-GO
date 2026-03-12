#!/bin/bash






if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi




python train.py \
    --model FGN \
    --data COVID \
    --feature_size 55\
    --embed_size 256 \
    --hidden_size 512 \
    --batch_size 4 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.0001 \
    --device cuda:0 >logs/LongForecasting/COVID_FGN.log


python train.py \
    --model SpikF \
    --data COVID \
    --feature_size 55\
    --embed_size 256 \
    --hidden_size 512 \
    --batch_size 4 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.0001 \
    --T 16 \
    --blocks 2\
    --device cuda:0 >logs/LongForecasting/COVID_SpikF.log



python train.py \
    --model iSpikformer \
    --data COVID \
    --feature_size 55\
    --embed_size 256 \
    --hidden_size 512 \
    --batch_size 4 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.0001 \
    --device cuda:0 >logs/LongForecasting/COVID_iSpikformer.log


python train.py \
    --model SpikF_GO \
    --data COVID \
    --feature_size 55\
    --embed_size 256 \
    --hidden_size 512 \
    --batch_size 4 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.0001 \
    --energy_loss True \
    --device cuda:0 >logs/LongForecasting/COVID_SpikFGO.log



python train.py \
    --model SpikF_GO_CPG \
    --data COVID \
    --feature_size 55\
    --embed_size 256 \
    --hidden_size 512 \
    --batch_size 4 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.0001 \
    --energy_loss True \
    --affine \
    --device cuda:0 >logs/LongForecasting/COVID_SpikF_GOCPG.log



python train.py \
    --model SpikeRNN_CPG \
    --data COVID \
    --feature_size 55\
    --embed_size 128 \
    --hidden_size 64\
    --batch_size 4 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.0001 \
    --blocks 1 \
    --device cuda:0 >logs/LongForecasting/COVID_SpikeRNNCPG.log


python train.py \
    --model SpikeGRU \
    --data COVID \
    --feature_size 55\
    --embed_size 256 \
    --hidden_size 64 \
    --batch_size 4 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.0001 \
    --blocks 3 \
    --device cuda:0 >logs/LongForecasting/COVID_SpikeGRU.log



python train.py \
    --model SpikeTCN_CPG \
    --data COVID \
    --feature_size 55\
    --embed_size 64 \
    --hidden_size 64\
    --batch_size 4 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.0001 \
    --blocks 3 \
    --device cuda:0 >logs/LongForecasting/COVID_SpikeTCNCPG.log



python train.py \
    --model Spikformer_CPG \
    --data COVID \
    --feature_size 55\
    --embed_size 128 \
    --hidden_size 128\
    --batch_size 4 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.0001 \
    --blocks 2\
    --device cuda:0 >logs/LongForecasting/COVID_SpikformerCPG.log



python train.py \
    --model TSGRU \
    --data COVID \
    --feature_size 55\
    --embed_size 128 \
    --hidden_size 128 \
    --batch_size 4 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.0001 \
    --device cuda:0 >logs/LongForecasting/COVID_TSGRU.log



python train.py \
    --model TSFormer \
    --data COVID \
    --feature_size 55\
    --embed_size 128 \
    --hidden_size 128\
    --batch_size 4 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.0001 \
    --block 2\
    --device cuda:0 >logs/LongForecasting/COVID_TSFormer.log


python train.py \
    --model TSTCN \
    --data COVID \
    --feature_size 55\
    --embed_size 128 \
    --hidden_size 128\
    --batch_size 4 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.0001 \
    --kernel_size 3\
    --blocks 3 \
    --device cuda:0 >logs/LongForecasting/COVID_TSTCN.log
