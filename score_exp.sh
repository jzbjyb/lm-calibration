#!/usr/bin/env bash

tpu=$1

# mc

# uq_clean_test uq_clean_train test uq_clean_train_inp uq_clean_test_inp uq_clean_test_bt uq_clean_test_ret
for task in uq_clean_test_ret uq_clean_train_inp uq_clean_test_inp; do
    for model in 3B; do
        if [[ $task == 'uq_sub_test' ]]; then
            output_root=output/exp/uq_sub_test/dev
            mix=uq_sub_test_mix
            split=dev
        elif [[ $task == 'uq_test' ]]; then
            output_root=output/exp/uq_test/dev
            mix=uq_test_mix
            split=dev
        elif [[ $task == 'uq_clean_test' ]]; then
            output_root=output/exp/uq_clean_test/dev
            mix=uq_clean_test_mix
            split=dev
        elif [[ $task == 'uq_clean_test_inp' ]]; then
            output_root=output/exp/uq_clean_test/dev/inp
            mix=uq_clean_test_inp_mix
            split=dev
        elif [[ $task == 'uq_clean_test_bt' ]]; then
            output_root=output/exp/uq_clean_test/dev/bt
            mix=uq_clean_test_bt_replace_mix
            split=dev
        elif [[ $task == 'uq_clean_test_ret' ]]; then
            output_root=output/exp/uq_clean_test/dev/ret
            mix=uq_clean_test_ret_drqa_3s_mix
            split=dev
        elif [[ $task == 'uq_clean_train' ]]; then
            output_root=output/exp/uq_clean_train/dev
            mix=uq_clean_train_mix
            split=dev
        elif [[ $task == 'uq_clean_train_inp' ]]; then
            output_root=output/exp/uq_clean_train/dev/inp
            mix=uq_clean_train_inp_mix
            split=dev
        elif [[ $task == 'test' ]]; then
            output_root=output/exp/test/test
            mix=test_mix
            split=test
        fi

        if [[ $model == '3B' ]]; then
            step=1103000
        elif [[ $model == '11B' ]]; then
            step=1115000
        fi

        ./score.sh $tpu ${output_root}/${model}/uq_ft_margin.txt ${model} unifiedqa/ft_models/${model}_margin $step $mix $split &> nohup.out
        ./score.sh $tpu ${output_root}/${model}/uq_ft_softmax.txt ${model} unifiedqa/ft_models/${model}_softmax $step $mix $split &> nohup.out
        ./score.sh $tpu ${output_root}/${model}/uq.txt ${model} unifiedqa/models/${model} $step $mix $split &> nohup.out
        ./score.sh $tpu ${output_root}/${model}/t5.txt ${model} t5-data/pretrained_models/${model} $step $mix $split &> nohup.out
    done
done
