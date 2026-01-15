#!/bin/bash
python scripts/01_train_ldm.py -b configs/diff-fundus.yml

python scripts/01_train_ldm.py -b configs/diff-oct.yml