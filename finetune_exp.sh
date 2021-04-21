#!/usr/bin/env bash

tpu=$1

./finetune.sh $tpu 11B 11B_mh noloss uq_mh_ol_mix train mh &> output/log/ft_11B_mh.out
./finetune.sh $tpu 11B 11B_mh_mh noloss uq_mh_mh_ol_mix train mh &> output/log/ft_11B_mh_mh.out

./finetune.sh $tpu 11B 11B_mh_lr1e3 noloss uq_mh_ol_mix train mh &> output/log/ft_11B_mh_lr1e3.out
./finetune.sh $tpu 11B 11B_mh_mh_lr1e3 noloss uq_mh_mh_ol_mix train mh &> output/log/ft_11B_mh_mh_lr1e3.out
./finetune.sh $tpu 11B 11B_mh__sm_lr1e3 noloss uq_mh__sm_ol_mix train mh &> output/log/ft_11B_mh__sm_lr1e3.out
./finetune.sh $tpu 11B 11B_mh_s_m_lr1e3 noloss uq_mh_s_m_ol_mix train mh &> output/log/ft_11B_mh_s_m_lr1e3.out

./finetune.sh $tpu 11B 11B_mh_mh_reducehop_lr1e3 noloss uq_mh_mh_reducehop_ol_mix train mh &> output/log/ft_11B_mh_mh_reducehop_lr1e3.out

./finetune.sh $tpu 11B 11B_mh_dev_lr1e3 noloss uq_mh_dev_ol_mix train mh &> output/log/11B_mh_dev_lr1e3.out

#./finetune.sh $tpu 3B 3B_softmax_nolennorm softmax uq_clean_train_ol_mix train mc &> output/log/ft_3B_mc_softmax3.out
#./finetune.sh $tpu 3B 3B_margin_nolennorm margin uq_clean_train_ol_mix train mc &> output/log/ft_3B_mc_margin3.out

#./finetune.sh $tpu 11B 11B_softmax_nolennorm softmax uq_clean_train_ol_mix train mc &> output/log/ft_11B_mc_softmax2.out
#./finetune.sh $tpu 11B 11B_margin_nolennorm margin uq_clean_train_ol_mix train mc &> output/log/ft_11B_mc_margin2.out

#./finetune.sh $tpu 3B 3B_ext_softmax softmax uq_ext_decode_train_ol_uq3B_mix train ext &> output/log/ft_3B_ext_softmax.out
#./finetune.sh $tpu 3B 3B_ext_margin margin uq_ext_decode_train_ol_uq3B_mix train ext &> output/log/ft_3B_ext_margin.out

#./finetune.sh $tpu 3B 3B_ext_dedup_softmax softmax uq_ext_decode_train_ol_uq3B_dedup_mix train ext &> output/log/ft_3B_ext_dedup_softmax.out
#./finetune.sh $tpu 3B 3B_ext_dedup_margin margin uq_ext_decode_train_ol_uq3B_dedup_mix train ext &> output/log/ft_3B_ext_dedup_margin.out

#./finetune.sh $tpu 3B 3B_ext_sample_softmax softmax uq_ext_decode_train_ol_uq3B_sample_mix train ext &> output/log/ft_3B_ext_sample_softmax.out
#./finetune.sh $tpu 3B 3B_ext_sample_margin margin uq_ext_decode_train_ol_uq3B_sample_mix train ext &> output/log/ft_3B_ext_sample_margin.out

#./finetune.sh $tpu 3B 3B_ext_span_softmax softmax uq_ext_decode_train_ol_uq3B_span_topk_mix train ext &> output/log/ft_3B_ext_span_softmax.out
#./finetune.sh $tpu 3B 3B_ext_span_margin margin uq_ext_decode_train_ol_uq3B_span_topk_mix train ext &> output/log/ft_3B_ext_span_margin.out
