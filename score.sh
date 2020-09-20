#!/usr/bin/env bash

model_dir=gs://neulab-qa/t5-data/pretrained_models/3B
#model_dir=gs://neulab-qa/unifiedqa/models/3B

./run_test.py \
    --tpu=default-dgdw2  \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${model_dir}" \
    --gin_file="gs://neulab-qa/t5-data/pretrained_models/3B/operative_config.gin" \
    --gin_file="score_from_file.gin" \
    --gin_file="greedy_decode.gin" \
    --gin_param="inputs_filename = 'test.prep.input/dev_input.txt'" \
    --gin_param="targets_filename = 'test.prep.input/dev_target.txt'" \
    --gin_param="scores_filename = 'output/dev_unifiedqa_score.txt'" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
    --gin_param="infer_checkpoint_step = 'all'"
