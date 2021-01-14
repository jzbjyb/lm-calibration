#!/usr/bin/env bash

set -e

testroot=output_moreep/exp/uq_clean_test/dev
trainroot=output_moreep/exp/uq_clean_train/dev
raw_cp=../unifiedqa/bart/out/gptlarge/best-model-045000.pt
sf_cp=../unifiedqa/bart/models/gpt2_ft_softmax.pt.10000
cp=../unifiedqa/bart/models/gpt2_ft_margin.pt.12000

CUDA_VISIBLE_DEVICES=1 python score_pytorch.py --data data/unifiedqa --domains clean_test_domains --out ${testroot}/gptlargeuq/gptlarge.txt &>> nohup.out &
CUDA_VISIBLE_DEVICES=2 python score_pytorch.py --data data/unifiedqa --domains clean_test_domains --out ${testroot}/gptlargeuq/gptlargeuq.txt --checkpoint ${raw_cp} &>> nohup.out &
CUDA_VISIBLE_DEVICES=3 python score_pytorch.py --data data/unifiedqa --domains clean_text_domains --out ${testroot}/gptlargeuq/gptlargeuq_ft_softmax.txt --checkpoint ${sf_cp} &>> nohup.out &
wait

CUDA_VISIBLE_DEVICES=1 python score_pytorch.py --data data/unifiedqa --domains clean_test_domains --out ${testroot}/gptlargeuq/gptlargeuq_ft_margin.txt --checkpoint ${cp} &>> nohup.out &
CUDA_VISIBLE_DEVICES=2 python score_pytorch.py --data data/unifiedqa --domains clean_train_domains --out ${trainroot}/gptlargeuq/gptlargeuq_ft_margin.txt --checkpoint ${cp} &>> nohup.out &
wait

CUDA_VISIBLE_DEVICES=1 python score_pytorch.py --data data/unifiedqa --domains clean_test_domains --out ${testroot}/inp/gptlargeuq/gptlargeuq_ft_margin.txt --use_inp --checkpoint ${cp} &>> nohup.out &
CUDA_VISIBLE_DEVICES=2 python score_pytorch.py --data data/unifiedqa --domains clean_train_domains --out ${trainroot}/inp/gptlargeuq/gptlargeuq_ft_margin.txt --use_inp --checkpoint ${cp} &>> nohup.out &
wait

CUDA_VISIBLE_DEVICES=1 python score_pytorch.py --data data/unifiedqa_ret_drqa_3s --domains clean_test_domains --out ${testroot}/ret/gptlargeuq/gptlargeuq_ft_margin.txt --has_ret --checkpoint ${cp} &>> nohup.out &
CUDA_VISIBLE_DEVICES=2 python score_pytorch.py --data data/unifiedqa_bt_dedup_replace --domains clean_test_domains --out ${testroot}/bt_dedup/gptlargeuq/gptlargeuq_ft_margin.txt --checkpoint ${cp} &>> nohup.out &
wait

CUDA_VISIBLE_DEVICES=1 python score_pytorch.py --data data/unifiedqa_ret_drqa_3s_bt_dedup_replace --domains clean_test_domains --out ${testroot}/ret_bt_dedup/gptlargeuq/gptlargeuq_ft_margin.txt --has_ret --checkpoint ${cp} &>> nohup.out &
CUDA_VISIBLE_DEVICES=2 python score_pytorch.py --data data/unifiedqa_ret_drqa_3s_bt_dedup_replace --domains clean_train_domains --out ${trainroot}/ret_bt_dedup/gptlargeuq/gptlargeuq_ft_margin.txt --has_ret --checkpoint ${cp} &>> nohup.out &
wait

CUDA_VISIBLE_DEVICES=1 python score_pytorch.py --data data/unifiedqa_ret_drqa_3s_bt_dedup_replace --domains clean_test_domains --out ${testroot}/ret_bt_dedup_inp/gptlargeuq/gptlargeuq_ft_margin.txt --has_ret --use_inp --checkpoint ${cp} &>> nohup.out &
CUDA_VISIBLE_DEVICES=2 python score_pytorch.py --data data/unifiedqa_ret_drqa_3s_bt_dedup_replace --domains clean_train_domains --out ${trainroot}/ret_bt_dedup_inp/gptlargeuq/gptlargeuq_ft_margin.txt --has_ret --use_inp --checkpoint ${cp} &>> nohup.ou