#!/usr/bin/env bash

# mc

for model_type in 3B; do
    # temp
    python train_cal.py --model temp --mix uq_clean_train_mix --split dev \
        --score output/exp/uq_clean_train/dev/${model_type}/uq_ft_margin.txt &> output/exp/model/temp/${model_type}_ft_margin.out
    # xgb
    python train_cal.py --model xgb --mix uq_clean_train_mix --split dev \
        --score output/exp/uq_clean_train/dev/${model_type}/uq_ft_margin.txt \
        --inp_perp output/exp/uq_clean_train/dev/inp/${model_type}/uq_ft_margin.txt \
        --out output/exp/model/xgb/${model_type}_ft_margin.bin &> output/exp/model/xgb/${model_type}_ft_margin.out
done
