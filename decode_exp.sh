#!/usr/bin/env bash

tpu=$1
model=3B

./score.sh $tpu output/exp/uq_ext/dev/${model}/uq.txt ${model} unifiedqa/models/${model} 1100500 uq_ext_mix dev
python prep.py first_decode
./score.sh $tpu output/exp/uq_ext_first/dev/${model}/uq.txt ${model} unifiedqa/models/${model} 1100500 uq_ext_first_decode_uq3B_mix dev

./score.sh $tpu output/exp/uq_ext/train/${model}/uq.txt ${model} unifiedqa/models/${model} 1100500 uq_ext_mix train
python prep.py first_decode
./score.sh $tpu output/exp/uq_ext_first/train/${model}/uq.txt ${model} unifiedqa/models/${model} 1100500 uq_ext_first_decode_uq3B_mix train
