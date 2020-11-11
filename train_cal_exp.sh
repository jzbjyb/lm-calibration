#!/usr/bin/env bash

# mc

for model_type in 3B; do
    for train_set in uq_clean_train uq_clean_test; do  # uq_clean_train uq_clean_test
        # temp
        python train_cal.py --model temp --mix ${train_set}_mix --split dev \
            --score output/exp/${train_set}/dev/${model_type}/uq_ft_margin.txt &> output/exp/model/temp/${model_type}_${train_set}_ft_margin.out

        # xgb
        python train_cal.py --model xgb --mix ${train_set}_mix --split dev \
            --score output/exp/${train_set}/dev/${model_type}/uq_ft_margin.txt \
            --inp_perp output/exp/${train_set}/dev/inp/${model_type}/uq_ft_margin.txt \
            --out output/exp/model/xgb/${model_type}_${train_set}_ft_margin.bin &> output/exp/model/xgb/${model_type}_${train_set}_ft_margin.out

        # ret bt
        if [[ $train_set == 'uq_clean_train' ]]; then
            reb_bt_mix=uq_clean_train_bt_mix
        elif [[ $train_set == 'uq_clean_test' ]]; then
            reb_bt_mix=uq_clean_test_bt_mix
        fi
        python train_cal.py --model temp --mix ${reb_bt_mix} --split dev \
            --score output/exp/${train_set}/dev/ret_bt/${model_type}/uq_ft_margin.txt &> output/exp/model/temp/${model_type}_${train_set}_ret_bt_ft_margin.out
        python train_cal.py --model xgb --mix ${reb_bt_mix} --split dev \
            --score output/exp/${train_set}/dev/ret_bt/${model_type}/uq_ft_margin.txt \
            --inp_perp output/exp/${train_set}/dev/ret_bt_inp/${model_type}/uq_ft_margin.txt \
            --out output/exp/model/xgb/${model_type}_${train_set}_ret_bt_ft_margin.bin &> output/exp/model/xgb/${model_type}_${train_set}_ret_bt_ft_margin.out
    done
done
