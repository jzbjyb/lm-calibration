#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python score_pytorch.py --data data/unifiedqa --domains clean_test_domains --out output/exp/uq_clean_test/dev/gptxl/gptxl_fs.txt --fewshot 3 --fewshot_data data/unifiedqa --fewshot_domains clean_test_domains &>> nohup.out &
CUDA_VISIBLE_DEVICES=2 python score_pytorch.py --data data/unifiedqa_ret_drqa_3s --domains clean_test_domains --out output/exp/uq_clean_test/dev/ret/gptxl/gptxl_fs.txt --has_ret --fewshot 3 --fewshot_data data/unifiedqa --fewshot_domains clean_test_domains &>> nohup.out &
CUDA_VISIBLE_DEVICES=3 python score_pytorch.py --data data/unifiedqa_bt_dedup_replace --domains clean_test_domains --out output/exp/uq_clean_test/dev/bt_dedup/gptxl/gptxl_fs.txt --fewshot 3 --fewshot_data data/unifiedqa --fewshot_domains clean_test_domains &>> nohup.out &
wait
