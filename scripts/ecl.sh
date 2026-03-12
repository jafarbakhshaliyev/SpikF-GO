#!/bin/bash






if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi








python train.py \
    --model FGN \
    --data electricity \
    --feature_size 370\
    --embed_size 128 \
    --hidden_size 256 \
    --batch_size 16 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --device cuda:0 >logs/LongForecasting/ECL_FGN.log


python train.py \
    --model SpikF \
    --data electricity \
    --feature_size 370\
    --embed_size 128 \
    --hidden_size 256 \
    --batch_size 16 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --T 16 \
    --blocks 2\
    --device cuda:0 >logs/LongForecasting/ECL_SpikF.log


python train.py \
    --model iSpikformer \
    --data electricity \
    --feature_size 370\
    --embed_size 128 \
    --hidden_size 256 \
    --batch_size 16 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --blocks 2 \
    --device cuda:0 >logs/LongForecasting/ECL_iSpikformer.log



python train.py \
    --model SpikF_GO \
    --data electricity \
    --feature_size 370\
    --embed_size 128 \
    --hidden_size 256 \
    --batch_size 16 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --energy_loss True \
    --device cuda:0 >logs/LongForecasting/ECL_SpikFGO.log


python train.py \
    --model SpikF_GO_CPG \
    --data electricity \
    --feature_size 370\
    --embed_size 128 \
    --hidden_size 256 \
    --batch_size 16 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --energy_loss True \
    --device cuda:0 >logs/LongForecasting/ECL_SpikFGOCPG.log


python train.py \
    --model SpikeRNN_CPG \
    --data electricity \
    --feature_size 370\
    --embed_size 128 \
    --hidden_size 128\
    --batch_size 16 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --blocks 2 \
    --device cuda:0 >logs/LongForecasting/ECL_SpikeRNNCPG.log




python train.py \
    --model SpikeGRU \
    --data electricity \
    --feature_size 370\
    --embed_size 128 \
    --hidden_size 64 \
    --batch_size 16 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --device cuda:0 >logs/LongForecasting/ECL_SpikeGRU.log




python train.py \
    --model SpikeTCN_CPG \
    --data electricity \
    --feature_size 370\
    --embed_size 128 \
    --hidden_size 64\
    --batch_size 16 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --blocks 3\
    --device cuda:0 >logs/LongForecasting/ECL_SpikeTCNCPG.log




python train.py \
    --model Spikformer_CPG \
    --data electricity \
    --feature_size 370\
    --embed_size 128 \
    --hidden_size 128\
    --batch_size 16 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --blocks 2 \
    --device cuda:0 >logs/LongForecasting/ECL_SpikformerCPG.log



python train.py \
    --model TSTCN \
    --data electricity \
    --feature_size 370\
    --embed_size 128 \
    --hidden_size 64 \
    --batch_size 16 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --kernel_size 3\
    --blocks 3 \
    --device cuda:0 >logs/LongForecasting/ECL_TSTCN.log


python train.py \
    --model TSGRU \
    --data electricity \
    --feature_size 370\
    --embed_size 128 \
    --hidden_size 64 \
    --batch_size 16 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --device cuda:0 >logs/LongForecasting/ECL_TSGRU.log




python train.py \
    --model TSFormer \
    --data electricity \
    --feature_size 370\
    --embed_size 128 \
    --hidden_size 64 \
    --batch_size 16 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 12 \
    --train_epochs 100 \
    --learning_rate 0.00001 \
    --device cuda:0 >logs/LongForecasting/ECL_TSFormer.log
