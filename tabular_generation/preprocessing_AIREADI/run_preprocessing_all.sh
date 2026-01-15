#!/bin/bash
DATA_PATH=/data/7TB/nick/ai_readi_v3/

python 1_align_raw_wearable_data.py --DATA_PATH $DATA_PATH
python 2_get_wearable_patient_days.py --DATA_PATH $DATA_PATH
python 3_get_ekg.py --DATA_PATH $DATA_PATH 
python 4_aggregate_data_modalities.py --DATA_PATH $DATA_PATH
python 5_preprocess_aireadi.py --SAVE_PATH /data/7TB/nick/ai_readi_v3/tabular_processed_data
