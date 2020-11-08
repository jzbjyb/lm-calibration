#!/usr/bin/env bash

tpu=$1

# mc

output_root=output/exp/uq_test/dev

./score.sh $tpu ${output_root}/3B/uq_ft_softmax.txt 3B unifiedqa/ft_models/3B_softmax 1105000 uq_test_mix dev &> nohup.out
./score.sh $tpu ${output_root}/3B/uq_ft_margin.txt 3B unifiedqa/ft_models/3B_margin 1105000 uq_test_mix dev &> nohup.out
./score.sh $tpu ${output_root}/3B/uq.txt 3B unifiedqa/models/3B 1100500 uq_test_mix dev &> nohup.out
./score.sh $tpu ${output_root}/3B/t5.txt 3B t5-data/pretrained_models/3B 1000000 uq_test_mix dev &> nohup.out
