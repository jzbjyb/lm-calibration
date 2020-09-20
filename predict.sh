#!/usr/bin/env bash

./run_test.py \
    --tpu=default-dgdw2  \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="gs://neulab-qa/t5-data/pretrained_models/3B" \
    --gin_file="gs://neulab-qa/t5-data/pretrained_models/3B/operative_config.gin" \
    --gin_file="infer.gin" \
    --gin_file="greedy_decode.gin" \
    --gin_param="input_filename = 'test.prep.input/dev.txt'" \
    --gin_param="output_filename = 'output/dev.txt'" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
    --gin_param="infer_checkpoint_step = 'all'"
