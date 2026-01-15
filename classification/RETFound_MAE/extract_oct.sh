#!/bin/bash

python main_finetune.py \
    --model RETFound_mae \
    --savemodel \
    --global_pool \
    --batch_size 64 \
    --world_size 1 \
    --epochs 50 \
    --seed 0 \
    --warmup_epochs 5 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.05 \
    --nb_classes 3 \
    --data_path_train /data/7TB/nick/synth_ai_readi_oct/  \
    --data_path_test /data/7TB/nick/synth_ai_readi_oct/  \
    --input_size 224 \
    --task dx_oct_synth \
    --finetune RETFound_mae_meh \
    --ckpt /home/nick/ai_readi_synth/classification/RETFound_MAE/output_dir/dx_oct_synth_0/checkpoint-best.pth \
    --modality oct \
    --synth_train \
    --synth_test \
    --eval \
    --extract


python main_finetune.py \
    --model RETFound_mae \
    --savemodel \
    --global_pool \
    --batch_size 64 \
    --world_size 1 \
    --epochs 50 \
    --seed 0 \
    --warmup_epochs 5 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.05 \
    --nb_classes 3 \
    --data_path_train /data/7TB/nick/ai_readi_v3/  \
    --data_path_test /data/7TB/nick/ai_readi_v3/  \
    --input_size 224 \
    --task dx_oct \
    --finetune RETFound_mae_meh \
    --ckpt /home/nick/ai_readi_synth/classification/RETFound_MAE/output_dir/dx_oct_synth_0/checkpoint-best.pth \
    --modality oct \
    --eval \
    --extract