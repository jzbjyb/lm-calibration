#!/usr/bin/env bash

tpu=$1

# mc

for task in uq_sub_test; do
    for model in 3B; do
        if [[ $task == 'uq_sub_test' ]]; then
            output_root=output/exp/uq_sub_test/dev
            mix=uq_sub_test_mix
            split=dev
        elif [[ $task == 'uq_test' ]]; then
            output_root=output/exp/uq_test/dev
            mix=uq_test_mix
            split=dev
        fi

        if [[ $model == '3B' ]]; then
            step=1103000
        elif [[ $model == '11B' ]]; then
            step=1115000
        fi

        ./score.sh $tpu ${output_root}/${model}/uq_ft_softmax.txt ${model} unifiedqa/ft_models/${model}_softmax $step $mix $split &> nohup.out
        ./score.sh $tpu ${output_root}/${model}/uq_ft_margin.txt ${model} unifiedqa/ft_models/${model}_margin $step $mix $split &> nohup.out
        ./score.sh $tpu ${output_root}/${model}/uq.txt ${model} unifiedqa/models/${model} $step $mix $split &> nohup.out
        ./score.sh $tpu ${output_root}/${model}/t5.txt ${model} t5-data/pretrained_models/${model} $step $mix $split &> nohup.out
    done
done
