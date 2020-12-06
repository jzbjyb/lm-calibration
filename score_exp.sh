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
# test_bt_dedup test_ret_bt_dedup test_ret_bt_dedup_inp

# uq_clean_test_bt_dedup_top10 uq_clean_test_bt_dedup_top20

for task in uq_clean_test uq_clean_train test uq_clean_test_inp uq_clean_test_bt_dedup uq_clean_test_ret uq_clean_test_ret_bt_dedup uq_clean_test_ret_bt_dedup_inp uq_clean_train_inp uq_clean_train_bt_dedup uq_clean_train_ret uq_clean_train_ret_bt_dedup uq_clean_train_ret_bt_dedup_inp test_inp test_bt_dedup test_ret test_ret_bt_dedup test_ret_bt_dedup_inp uq_clean_test_bt_dedup_top20; do
    for model in 3B; do
        if [[ $task == 'uq_clean_test' ]]; then
            output_root=output/exp/uq_clean_test/dev_nolennorm
            mix=uq_clean_test_mix
            split=dev
        elif [[ $task == 'uq_clean_test_inp' ]]; then
            output_root=output/exp/uq_clean_test/dev_nolennorm/inp
            mix=uq_clean_test_inp_mix
            split=dev
        elif [[ $task == 'uq_clean_test_bt_dedup' ]]; then
            output_root=output/exp/uq_clean_test/dev_nolennorm/bt_dedup
            mix=uq_clean_test_bt_dedup_replace_mix
            split=dev
        elif [[ $task == 'uq_clean_test_bt_dedup_top20' ]]; then
            output_root=output/exp/uq_clean_test/dev_nolennorm/bt_dedup_top20
            mix=uq_clean_test_bt_dedup_top20_replace_mix
            split=dev
        elif [[ $task == 'uq_clean_test_ret' ]]; then
            output_root=output/exp/uq_clean_test/dev_nolennorm/ret
            mix=uq_clean_test_ret_drqa_3s_mix
            split=dev
        elif [[ $task == 'uq_clean_test_ret_bt_dedup' ]]; then
            output_root=output/exp/uq_clean_test/dev_nolennorm/ret_bt_dedup
            mix=uq_clean_test_ret_drqa_3s_bt_dedup_replace_mix
            split=dev
        elif [[ $task == 'uq_clean_test_ret_bt_dedup_inp' ]]; then
            output_root=output/exp/uq_clean_test/dev_nolennorm/ret_bt_dedup_inp
            mix=uq_clean_test_ret_drqa_3s_bt_dedup_replace_inp_mix
            split=dev
        elif [[ $task == 'uq_clean_train' ]]; then
            output_root=output/exp/uq_clean_train/dev_nolennorm
            mix=uq_clean_train_mix
            split=dev
        elif [[ $task == 'uq_clean_train_inp' ]]; then
            output_root=output/exp/uq_clean_train/dev_nolennorm/inp
            mix=uq_clean_train_inp_mix
            split=dev
        elif [[ $task == 'uq_clean_train_bt_dedup' ]]; then
            output_root=output/exp/uq_clean_train/dev_nolennorm/bt_dedup
            mix=uq_clean_train_bt_dedup_replace_mix
            split=dev
        elif [[ $task == 'uq_clean_train_ret' ]]; then
            output_root=output/exp/uq_clean_train/dev_nolennorm/ret
            mix=uq_clean_train_ret_drqa_3s_mix
            split=dev
        elif [[ $task == 'uq_clean_train_ret_bt_dedup' ]]; then
            output_root=output/exp/uq_clean_train/dev_nolennorm/ret_bt_dedup
            mix=uq_clean_train_ret_drqa_3s_bt_dedup_replace_mix
            split=dev
        elif [[ $task == 'uq_clean_train_ret_bt_dedup_inp' ]]; then
            output_root=output/exp/uq_clean_train/dev_nolennorm/ret_bt_dedup_inp
            mix=uq_clean_train_ret_drqa_3s_bt_dedup_replace_inp_mix
            split=dev
        elif [[ $task == 'test' ]]; then
            output_root=output/exp/test/test_nolennorm
            mix=test_mix
            split=test
        elif [[ $task == 'test_inp' ]]; then
            output_root=output/exp/test/test_nolennorm/inp
            mix=test_inp_mix
            split=test
        elif [[ $task == 'test_bt_dedup' ]]; then
            output_root=output/exp/test/test_nolennorm/bt_dedup
            mix=test_bt_dedup_replace_mix
            split=test
        elif [[ $task == 'test_ret' ]]; then
            output_root=output/exp/test/test_nolennorm/ret
            mix=test_ret_drqa_3s_mix
            split=test
        elif [[ $task == 'test_ret_bt_dedup' ]]; then
            output_root=output/exp/test/test_nolennorm/ret_bt_dedup
            mix=test_ret_drqa_3s_bt_dedup_replace_mix
            split=test
        elif [[ $task == 'test_ret_bt_dedup_inp' ]]; then
            output_root=output/exp/test/test_nolennorm/ret_bt_dedup_inp
            mix=test_ret_drqa_3s_bt_dedup_replace_inp_mix
            split=test
        fi

        if [[ $model == '3B' ]]; then
            step=1103000
        elif [[ $model == '11B' ]]; then
            step=1115000
        fi

        ./score.sh $tpu ${output_root}/${model}/uq_ft_margin.txt ${model} unifiedqa/ft_models/${model}_margin_nolennorm $step $mix $split &> nohup.out
        if [[ $task == 'uq_clean_test' ]] || [[ $task == 'uq_clean_train' ]] || [[ $task == 'test' ]]; then
            ./score.sh $tpu ${output_root}/${model}/uq_ft_softmax.txt ${model} unifiedqa/ft_models/${model}_softmax_nolennorm $step $mix $split &> nohup.out
            #./score.sh $tpu ${output_root}/${model}/uq.txt ${model} unifiedqa/models/${model} 1100500 $mix $split &> nohup.out
            #./score.sh $tpu ${output_root}/${model}/t5.txt ${model} t5-data/pretrained_models/${model} 1000000 $mix $split &> nohup.out
        fi
    done
done


# ext

# uq_ext_train uq_ext_test
# uq_ext_train_inp uq_ext_test_inp
# uq_ext_train_ret uq_ext_test_ret
# uq_ext_train_bt uq_ext_test_bt
# uq_ext_train_ret_bt uq_ext_test_ret_bt
# uq_ext_train_ret_bt_inp uq_ext_test_ret_bt_inp

# uq_ext uq_ext_first
# uq_ext_first_topk uq_ext_first_topk_bt

suffix='_sample'
for task in uq_ext_first_topk uq_ext_first_topk_bt; do
    for model in 3B; do
        if [[ $task == 'uq_ext' ]]; then
            output_root=output/exp/uq_ext/dev
            mix=uq_ropes_oc_mix
            split=dev
        elif [[ $task == 'uq_ext_first' ]]; then
            output_root=output/exp/uq_ext_first/dev
            mix=uq_ropes_first_decode_uq3B_mix
            split=dev
        elif [[ $task == 'uq_ext_first_topk' ]]; then
            output_root=output/exp/uq_ext_first_topk/dev
            mix=uq_ropes_first_decode_topk_uq3B_mix
            split=dev
        elif [[ $task == 'uq_ext_first_topk_bt' ]]; then
            output_root=output/exp/uq_ext_first_topk/dev/bt_dedup
            mix=uq_ropes_first_decode_topk_uq3B_bt_dedup_mix
            split=dev
        elif [[ $task == 'uq_ext_train' ]]; then
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

        ./score.sh $tpu ${output_root}/${model}/uq.txt ${model} unifiedqa/models/${model} 1100500 $mix $split &> nohup.out
        if [[ $task == 'uq_ext_train' ]] || [[ $task == 'uq_ext_test' ]]; then
            ./score.sh $tpu ${output_root}/${model}/uq_ft_margin.txt ${model} unifiedqa/ft_models/${model}_ext_sample_margin $step $mix $split &> nohup.out
            ./score.sh $tpu ${output_root}/${model}/uq_ft_softmax.txt ${model} unifiedqa/ft_models/${model}_ext_sample_softmax $step $mix $split &> nohup.out
            ./score.sh $tpu ${output_root}/${model}/t5.txt ${model} t5-data/pretrained_models/${model} 1000000 $mix $split &> nohup.out
        fi
    done
done
