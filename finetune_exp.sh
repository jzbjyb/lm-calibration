#!/usr/bin/env bash

tpu=$1

#./finetune.sh $tpu 3B 3B_softmax_nolennorm softmax uq_clean_train_ol_mix train mc &> output/log/ft_3B_mc_softmax3.out
#./finetune.sh $tpu 3B 3B_margin_nolennorm margin uq_clean_train_ol_mix train mc &> output/log/ft_3B_mc_margin3.out

./finetune.sh $tpu 11B 11B_softmax_nolennorm softmax uq_clean_train_ol_mix train mc &> output/log/ft_11B_mc_softmax2.out
./finetune.sh $tpu 11B 11B_margin_nolennorm margin uq_clean_train_ol_mix train mc &> output/log/ft_11B_mc_margin2.out

#./finetune.sh $tpu 3B 3B_ext_softmax softmax uq_ext_decode_train_ol_uq3B_mix train ext &> output/log/ft_3B_ext_softmax.out
#./finetune.sh $tpu 3B 3B_ext_margin margin uq_ext_decode_train_ol_uq3B_mix train ext &> output/log/ft_3B_ext_margin.out

#./finetune.sh $tpu 3B 3B_ext_dedup_softmax softmax uq_ext_decode_train_ol_uq3B_dedup_mix train ext &> output/log/ft_3B_ext_dedup_softmax.out
#./finetune.sh $tpu 3B 3B_ext_dedup_margin margin uq_ext_decode_train_ol_uq3B_dedup_mix train ext &> output/log/ft_3B_ext_dedup_margin.out

#./finetune.sh $tpu 3B 3B_ext_sample_softmax softmax uq_ext_decode_train_ol_uq3B_sample_mix train ext &> output/log/ft_3B_ext_sample_softmax.out
#./finetune.sh $tpu 3B 3B_ext_sample_margin margin uq_ext_decode_train_ol_uq3B_sample_mix train ext &> output/log/ft_3B_ext_sample_margin.out

#./finetune.sh $tpu 3B 3B_ext_span_softmax softmax uq_ext_decode_train_ol_uq3B_span_topk_mix train ext &> output/log/ft_3B_ext_span_softmax.out
#./finetune.sh $tpu 3B 3B_ext_span_margin margin uq_ext_decode_train_ol_uq3B_span_topk_mix train ext &> output/log/ft_3B_ext_span_margin.out
