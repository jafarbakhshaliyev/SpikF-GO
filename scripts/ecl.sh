#!/bin/bash






if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi




python maints_pred1.py \
    --model FGN \
    --data COVID \
    --feature_size 55\
    --embed_size 256 \
    --hidden_size 512 \
    --batch_size 4 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --seq_length 12 \
    --pre_length 48 \
    --train_epochs 100 \
    --learning_rate 0.0001 \
    --T 4 \
    --tau 2.0 \
    --alpha 1.0 \
    --device cuda:0 >logs/LongForecasting/COVID_48_FGN.log







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
    --device cuda:0 >logs/LongForecasting/ECL_SpikF.log



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
    --blocks 3 \
    --device cuda:0 >logs/LongForecasting/ECL_TSTCN.log
