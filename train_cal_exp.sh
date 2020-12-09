#!/usr/bin/env bash

# mc
: '
for model_type in 3B; do
    for train_set in uq_clean_train uq_clean_test; do
        # temp
        python train_cal.py --model temp --mix ${train_set}_mix --split dev \
            --score output/exp/${train_set}/dev_nolennorm/${model_type}/uq_ft_margin.txt &> output/exp/model_nolennorm/temp/${model_type}_${train_set}_ft_margin.out

        # xgb
        python train_cal.py --model xgb --mix ${train_set}_mix --split dev \
            --score output/exp/${train_set}/dev_nolennorm/${model_type}/uq_ft_margin.txt \
            --inp_perp output/exp/${train_set}/dev_nolennorm/inp/${model_type}/uq_ft_margin.txt \
            --out output/exp/model_nolennorm/xgb/${model_type}_${train_set}_ft_margin.bin &> output/exp/model_nolennorm/xgb/${model_type}_${train_set}_ft_margin.out
        # ret bt
        if [[ $train_set == 'uq_clean_train' ]]; then
            reb_bt_mix=uq_clean_train_ret_drqa_3s_bt_dedup_replace_mix
        elif [[ $train_set == 'uq_clean_test' ]]; then
            reb_bt_mix=uq_clean_test_ret_drqa_3s_bt_dedup_replace_mix
        fi
        suffix=ret_bt_dedup
        python train_cal.py --model temp --mix ${reb_bt_mix} --split dev \
            --score output/exp/${train_set}/dev_nolennorm/${suffix}/${model_type}/uq_ft_margin.txt &> output/exp/model_nolennorm/temp/${model_type}_${train_set}_${suffix}_ft_margin.out
        python train_cal.py --model xgb --mix ${reb_bt_mix} --split dev \
            --score output/exp/${train_set}/dev_nolennorm/${suffix}/${model_type}/uq_ft_margin.txt \
            --inp_perp output/exp/${train_set}/dev_nolennorm/${suffix}_inp/${model_type}/uq_ft_margin.txt \
            --out output/exp/model_nolennorm/xgb/${model_type}_${train_set}_${suffix}_ft_margin.bin &> output/exp/model_nolennorm/xgb/${model_type}_${train_set}_${suffix}_ft_margin.out
    done
done
'
# ext

for model_type in 3B; do
    for train_set in uq_ext_train_span_topk_nogold uq_ext_test_span_topk_nogold; do
        if [[ $train_set == 'uq_ext_train' ]]; then
            mix=uq_ext_decode_train_uq3B_mix
        elif [[ $train_set == 'uq_ext_test' ]]; then
            mix=uq_ext_decode_test_uq3B_mix
        elif [[ $train_set == 'uq_ext_train_dedup' ]]; then
            mix=uq_ext_decode_train_uq3B_dedup_mix
        elif [[ $train_set == 'uq_ext_test_dedup' ]]; then
            mix=uq_ext_decode_test_uq3B_dedup_mix
        elif [[ $train_set == 'uq_ext_train_span_topk_nogold' ]]; then
            mix=uq_ext_decode_train_uq3B_span_topk_nogold_mix
        elif [[ $train_set == 'uq_ext_test_span_topk_nogold' ]]; then
            mix=uq_ext_decode_test_uq3B_span_topk_nogold_mix
        fi

        # temp
        python train_cal.py --model temp --mix ${mix} --split dev \
            --score output/exp/${train_set}/dev/${model_type}/uq.txt &> output/exp/model/temp/${model_type}_${train_set}.out

        # xgb
        python train_cal.py --model xgb --mix ${mix} --split dev \
            --score output/exp/${train_set}/dev/${model_type}/uq.txt \
            --inp_perp output/exp/${train_set}/dev/inp/${model_type}/uq.txt \
            --out output/exp/model/xgb/${model_type}_${train_set}.bin &> output/exp/model/xgb/${model_type}_${train_set}.out

        # ret bt
        if [[ $train_set == 'uq_ext_train_dedup' ]]; then
            reb_bt_mix=uq_ext_decode_train_uq3B_dedup_ret_drqa_3s_bt_mix
        elif [[ $train_set == 'uq_ext_test_dedup' ]]; then
            reb_bt_mix=uq_ext_decode_test_uq3B_dedup_ret_drqa_3s_bt_mix
        elif [[ $train_set == 'uq_ext_train_span_topk_nogold' ]]; then
            reb_bt_mix=uq_ext_decode_train_uq3B_span_topk_nogold_ret_drqa_3s_bt_mix
        elif [[ $train_set == 'uq_ext_test_span_topk_nogold' ]]; then
            reb_bt_mix=uq_ext_decode_test_uq3B_span_topk_nogold_ret_drqa_3s_bt_mix
        fi
        suffix=ret_bt
        python train_cal.py --model temp --mix ${mix} --split dev \
            --score output/exp/${train_set}/dev/${suffix}/${model_type}/uq.txt &> output/exp/model/temp/${model_type}_${train_set}_${suffix}.out

    done
done
