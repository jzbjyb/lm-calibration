#!/usr/bin/env bash

gin_model_dir=gs://neulab-qa/t5-data/pretrained_models/small
model_dir=gs://neulab-qa/t5-data/pretrained_models/small_ft
#model_dir=gs://neulab-qa/unifiedqa/models/3B

./run_test.py \
    --tpu=default-dgdw2 \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${model_dir}" \
    --gin_file="${gin_model_dir}/operative_config.gin" \
    --t5_tfds_data_dir="${DATA_DIR}" \
    --gin_file="dataset.gin" \
    --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
    --gin_param="MIXTURE_NAME = 'multi_test_mix'" \
    --gin_param="run.train_steps = 100100"