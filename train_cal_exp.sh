#!/usr/bin/env bash

# mc

python train_cal.py --model temp --mix uq_clean_train_mix --split dev \
    --score output/exp/uq_clean_train/dev/3B/uq_ft_margin.txt

python train_cal.py --model xgb --mix uq_clean_train_mix --split dev \
    --score output/exp/uq_clean_train/dev/3B/uq_ft_margin.txt \
    --inp_perp output/exp/uq_clean_train/dev/inp/3B/uq_ft_margin.txt \
    --out output/exp/xgb/uq_inp_prep.bin
