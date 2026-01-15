#!/bin/bash

for seed in {0..4}
do
    # Train
    python main_finetune.py \
        --model RETFound_mae \
        --savemodel \
        --global_pool \
        --batch_size 64 \
        --world_size 1 \
        --epochs 30 \
        --seed $seed \
        --warmup_epochs 3 \
        --blr 5e-3 --layer_decay 0.65 \
        --weight_decay 0.05 --drop_path 0.05 \
        --nb_classes 3 \
        --data_path_train /data/7TB/nick/ai_readi_v3/  \
        --data_path_test /data/7TB/nick/ai_readi_v3/  \
        --input_size 224 \
        --task dx_fundus \
        --finetune RETFound_mae_meh \
        --modality fundus

    # Test
    python main_finetune.py \
        --model RETFound_mae \
        --savemodel \
        --eval \
        --global_pool \
        --batch_size 64 \
        --world_size 1 \
        --epochs 30 \
        --seed $seed \
        --warmup_epochs 3 \
        --blr 5e-3 --layer_decay 0.65 \
        --weight_decay 0.05 --drop_path 0.05 \
        --nb_classes 3 \
        --data_path_train /data/7TB/nick/ai_readi_v3/  \
        --data_path_test /data/7TB/nick/ai_readi_v3/  \
        --input_size 224 \
        --task dx_fundus \
        --ckpt output_dir/dx_fundus_"$seed"/checkpoint-best.pth \
        --modality fundus

    # # R2S
    # python main_finetune.py \
    #     --model RETFound_mae \
    #     --savemodel \
    #     --eval \
    #     --global_pool \
    #     --batch_size 64 \
    #     --world_size 1 \
    #     --epochs 30 \
    #     --seed $seed \
    #     --warmup_epochs 3 \
    #     --blr 5e-3 --layer_decay 0.65 \
    #     --weight_decay 0.05 --drop_path 0.05 \
    #     --nb_classes 3 \
    #     --data_path_train /data/7TB/nick/ai_readi_v3/  \
    #     --data_path_test /data/7TB/nick/synth_ai_readi_oct/  \
    #     --input_size 224 \
    #     --task dx_fundus_r2s \
    #     --ckpt output_dir/dx_fundus_"$seed"/checkpoint-best.pth \
    #     --modality fundus \
    #     --synth_test

done
