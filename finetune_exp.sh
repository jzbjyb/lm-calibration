#!/usr/bin/env bash

tpu=$1

./finetune.sh $tpu 3B 3B_softmax softmax uq_clean_train_ol_mix train mc &> output/log/ft_3B_mc_softmax2.out
./finetune.sh $tpu 3B 3B_margin margin uq_clean_train_ol_mix train mc &> output/log/ft_3B_mc_margin2.out

#./finetune.sh $tpu 11B 11B_softmax softmax uq_clean_train_ol_mix train mc &> output/log/ft_11B_mc_softmax.out
#./finetune.sh $tpu 11B 11B_margin margin uq_clean_train_ol_mix train mc &> output/log/ft_11B_mc_margin.out
