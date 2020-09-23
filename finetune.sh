#!/usr/bin/env bash

gin_model_dir=gs://neulab-qa/t5-data/pretrained_models/3B
#from_model_dir=gs://neulab-qa/t5-data/pretrained_models/3B/model.ckpt-1000000
#to_model_dir=gs://neulab-qa/t5-data/pretrained_models/3B_ft
from_model=gs://neulab-qa/unifiedqa/models/3B/model.ckpt-1100500
to_model_dir=gs://neulab-qa/unifiedqa/models/3B_ft_ind

./run_test.py \
    --tpu=default-dgdw2 \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${to_model_dir}" \
    --gin_file="${gin_model_dir}/operative_config.gin" \
    --t5_tfds_data_dir="${DATA_DIR}" \
    --gin_file="dataset.gin" \
    --gin_param="run.init_checkpoint = '${from_model}'" \
    --gin_param="utils.tpu_mesh_shape.model_parallelism = 8" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
    --gin_param="MIXTURE_NAME = 'uq_train_mix'" \
    --gin_param="run.train_steps = 1105000" \
    --gin_param="build_uq.neg_method = 'indicator'" \
    --gin_param="run.batch_size = ('tokens_per_batch', 16384)"
