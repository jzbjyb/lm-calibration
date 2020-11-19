#!/usr/bin/env bash

tpu=$1

# mc

# uq_clean_test uq_clean_train test
# uq_clean_train_inp uq_clean_test_inp test_inp
# uq_clean_test_bt uq_clean_test_ret uq_clean_test_ret_bt uq_clean_test_ret_bt_inp
# uq_clean_train_bt uq_clean_train_ret uq_clean_train_ret_bt uq_clean_train_ret_bt_inp
# test_bt test_ret test_ret_bt test_ret_bt_inp

# uq_clean_test_bt_dedup uq_clean_test_ret_bt_dedup uq_clean_test_ret_bt_dedup_inp
# uq_clean_train_bt_dedup uq_clean_train_ret_bt_dedup uq_clean_train_ret_bt_dedup_inp

: '
for task in uq_clean_test_bt_dedup uq_clean_test_ret_bt_dedup uq_clean_test_ret_bt_dedup_inp uq_clean_train_ret_bt_dedup_inp uq_clean_train_bt_dedup uq_clean_train_ret_bt_dedup; do
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
        elif [[ $task == 'uq_clean_test_bt_dedup' ]]; then
            output_root=output/exp/uq_clean_test/dev/bt_dedup
            mix=uq_clean_test_bt_dedup_replace_mix
            split=dev
        elif [[ $task == 'uq_clean_test_ret' ]]; then
            output_root=output/exp/uq_clean_test/dev/ret
            mix=uq_clean_test_ret_drqa_3s_mix
            split=dev
        elif [[ $task == 'uq_clean_test_ret_bt' ]]; then
            output_root=output/exp/uq_clean_test/dev/ret_bt
            mix=uq_clean_test_ret_drqa_3s_bt_replace_mix
            split=dev
        elif [[ $task == 'uq_clean_test_ret_bt_dedup' ]]; then
            output_root=output/exp/uq_clean_test/dev/ret_bt_dedup
            mix=uq_clean_test_ret_drqa_3s_bt_dedup_replace_mix
            split=dev
        elif [[ $task == 'uq_clean_test_ret_bt_inp' ]]; then
            output_root=output/exp/uq_clean_test/dev/ret_bt_inp
            mix=uq_clean_test_ret_drqa_3s_bt_replace_inp_mix
            split=dev
        elif [[ $task == 'uq_clean_test_ret_bt_dedup_inp' ]]; then
            output_root=output/exp/uq_clean_test/dev/ret_bt_dedup_inp
            mix=uq_clean_test_ret_drqa_3s_bt_dedup_replace_inp_mix
            split=dev
        elif [[ $task == 'uq_clean_train' ]]; then
            output_root=output/exp/uq_clean_train/dev
            mix=uq_clean_train_mix
            split=dev
        elif [[ $task == 'uq_clean_train_inp' ]]; then
            output_root=output/exp/uq_clean_train/dev/inp
            mix=uq_clean_train_inp_mix
            split=dev
        elif [[ $task == 'uq_clean_train_bt' ]]; then
            output_root=output/exp/uq_clean_train/dev/bt
            mix=uq_clean_train_bt_replace_mix
            split=dev
        elif [[ $task == 'uq_clean_train_bt_dedup' ]]; then
            output_root=output/exp/uq_clean_train/dev/bt_dedup
            mix=uq_clean_train_bt_dedup_replace_mix
            split=dev
        elif [[ $task == 'uq_clean_train_ret' ]]; then
            output_root=output/exp/uq_clean_train/dev/ret
            mix=uq_clean_train_ret_drqa_3s_mix
            split=dev
        elif [[ $task == 'uq_clean_train_ret_bt' ]]; then
            output_root=output/exp/uq_clean_train/dev/ret_bt
            mix=uq_clean_train_ret_drqa_3s_bt_replace_mix
            split=dev
        elif [[ $task == 'uq_clean_train_ret_bt_dedup' ]]; then
            output_root=output/exp/uq_clean_train/dev/ret_bt_dedup
            mix=uq_clean_train_ret_drqa_3s_bt_dedup_replace_mix
            split=dev
        elif [[ $task == 'uq_clean_train_ret_bt_inp' ]]; then
            output_root=output/exp/uq_clean_train/dev/ret_bt_inp
            mix=uq_clean_train_ret_drqa_3s_bt_replace_inp_mix
            split=dev
        elif [[ $task == 'uq_clean_train_ret_bt_dedup_inp' ]]; then
            output_root=output/exp/uq_clean_train/dev/ret_bt_dedup_inp
            mix=uq_clean_train_ret_drqa_3s_bt_dedup_replace_inp_mix
            split=dev
        elif [[ $task == 'test' ]]; then
            output_root=output/exp/test/test
            mix=test_mix
            split=test
        elif [[ $task == 'test_inp' ]]; then
            output_root=output/exp/test/test/inp
            mix=test_inp_mix
            split=test
        elif [[ $task == 'test_bt' ]]; then
            output_root=output/exp/test/test/bt
            mix=test_bt_replace_mix
            split=test
        elif [[ $task == 'test_ret' ]]; then
            output_root=output/exp/test/test/ret
            mix=test_ret_drqa_3s_mix
            split=test
        elif [[ $task == 'test_ret_bt' ]]; then
            output_root=output/exp/test/test/ret_bt
            mix=test_ret_drqa_3s_bt_replace_mix
            split=test
        elif [[ $task == 'test_ret_bt_inp' ]]; then
            output_root=output/exp/test/test/ret_bt_inp
            mix=test_ret_drqa_3s_bt_replace_inp_mix
            split=test
        fi

        if [[ $model == '3B' ]]; then
            step=1103000
        elif [[ $model == '11B' ]]; then
            step=1115000
        fi

        ./score.sh $tpu ${output_root}/${model}/uq_ft_margin.txt ${model} unifiedqa/ft_models/${model}_margin $step $mix $split &> nohup.out
        if [[ $task == 'uq_clean_test' ]] || [[ $task == 'uq_clean_train' ]] || [[ $task == 'test' ]]; then
            ./score.sh $tpu ${output_root}/${model}/uq_ft_softmax.txt ${model} unifiedqa/ft_models/${model}_softmax $step $mix $split &> nohup.out
            ./score.sh $tpu ${output_root}/${model}/uq.txt ${model} unifiedqa/models/${model} $step $mix $split &> nohup.out
            ./score.sh $tpu ${output_root}/${model}/t5.txt ${model} t5-data/pretrained_models/${model} $step $mix $split &> nohup.out
        fi
    done
done
'

# ext

# uq_ext_train uq_ext_test
# uq_ext_train_inp uq_ext_test_inp
# uq_ext_train_ret uq_ext_test_ret
# uq_ext_train_bt uq_ext_test_bt
# uq_ext_train_ret_bt uq_ext_test_ret_bt
# uq_ext_train_ret_bt_inp uq_ext_test_ret_bt_inp

suffix='_sample'
for task in uq_ext_test uq_ext_test_ret uq_ext_train uq_ext_train_ret uq_ext_test_inp uq_ext_train_inp; do
    for model in 3B; do
        if [[ $task == 'uq_ext_train' ]]; then
            output_root=output/exp/uq_ext_train${suffix}/dev
            mix=uq_ext_decode_train_uq3B${suffix}_mix
            split=dev
        elif [[ $task == 'uq_ext_train_inp' ]]; then
            output_root=output/exp/uq_ext_train${suffix}/dev/inp
            mix=uq_ext_decode_train_uq3B${suffix}_inp_mix
            split=dev
        elif [[ $task == 'uq_ext_train_ret' ]]; then
            output_root=output/exp/uq_ext_train${suffix}/dev/ret
            mix=uq_ext_decode_train_uq3B${suffix}_ret_drqa_3s_mix
            split=dev
        elif [[ $task == 'uq_ext_train_bt' ]]; then
            output_root=output/exp/uq_ext_train${suffix}/dev/bt
            mix=uq_ext_decode_train_uq3B${suffix}_bt_mix
            split=dev
        elif [[ $task == 'uq_ext_train_ret_bt' ]]; then
            output_root=output/exp/uq_ext_train${suffix}/dev/ret_bt
            mix=uq_ext_decode_train_uq3B${suffix}_ret_drqa_3s_bt_mix
            split=dev
        elif [[ $task == 'uq_ext_train_ret_bt_inp' ]]; then
            output_root=output/exp/uq_ext_train${suffix}/dev/ret_bt_inp
            mix=uq_ext_decode_train_uq3B${suffix}_ret_drqa_3s_bt_inp_mix
            split=dev
        elif [[ $task == 'uq_ext_test' ]]; then
            output_root=output/exp/uq_ext_test${suffix}/dev
            mix=uq_ext_decode_test_uq3B${suffix}_mix
            split=dev
        elif [[ $task == 'uq_ext_test_inp' ]]; then
            output_root=output/exp/uq_ext_test${suffix}/dev/inp
            mix=uq_ext_decode_test_uq3B${suffix}_inp_mix
            split=dev
        elif [[ $task == 'uq_ext_test_ret' ]]; then
            output_root=output/exp/uq_ext_test${suffix}/dev/ret
            mix=uq_ext_decode_test_uq3B${suffix}_ret_drqa_3s_mix
            split=dev
        elif [[ $task == 'uq_ext_test_bt' ]]; then
            output_root=output/exp/uq_ext_test${suffix}/dev/bt
            mix=uq_ext_decode_test_uq3B${suffix}_bt_mix
            split=dev
        elif [[ $task == 'uq_ext_test_ret_bt' ]]; then
            output_root=output/exp/uq_ext_test${suffix}/dev/ret_bt
            mix=uq_ext_decode_test_uq3B${suffix}_ret_drqa_3s_bt_mix
            split=dev
        elif [[ $task == 'uq_ext_test_ret_bt_inp' ]]; then
            output_root=output/exp/uq_ext_test${suffix}/dev/ret_bt_inp
            mix=uq_ext_decode_test_uq3B${suffix}_ret_drqa_3s_bt_inp_mix
            split=dev
        fi

        if [[ $model == '3B' ]]; then
            step=1103000
        elif [[ $model == '11B' ]]; then
            step=1115000
        fi

        ./score.sh $tpu ${output_root}/${model}/uq.txt ${model} unifiedqa/models/${model} $step $mix $split &> nohup.out
        if [[ $task == 'uq_ext_train' ]] || [[ $task == 'uq_ext_test' ]]; then
            ./score.sh $tpu ${output_root}/${model}/uq_ft_margin.txt ${model} unifiedqa/ft_models/${model}_ext_dedup_margin $step $mix $split &> nohup.out
            ./score.sh $tpu ${output_root}/${model}/uq_ft_softmax.txt ${model} unifiedqa/ft_models/${model}_ext_dedup_softmax $step $mix $split &> nohup.out
            ./score.sh $tpu ${output_root}/${model}/t5.txt ${model} t5-data/pretrained_models/${model} $step $mix $split &> nohup.out
        fi
    done
done

