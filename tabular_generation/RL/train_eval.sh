#!/bin/bash

python main.py \
        --OUT_DIR /data/7TB/nick/ai_readi_v3/tabular_processed_data/\
        --RESULT_CSV /home/nick/ai_readi_synth/tabular_generation/results.csv \
        --RUN_NAME AI_READI\
        --SEED 5 \
        --ITERS 30000 \
        --DATA_PATH /data/7TB/nick/ai_readi_v3/tabular_processed_data/ \
        --TRAIN

for seed in {0..4}
do
    python main.py \
        --OUT_DIR /data/7TB/nick/ai_readi_v3/tabular_processed_data/\
        --RESULT_CSV /home/nick/ai_readi_synth/tabular_generation/results.csv \
        --RUN_NAME AI_READI\
        --SEED $seed \
        --ITERS 30000 \
        --DATA_PATH /data/7TB/nick/ai_readi_v3/tabular_processed_data/
done
