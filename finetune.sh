#!/usr/bin/env bash

neg_method=$1
output=$2

tpu_name=default-dgdw2
gin_model_dir=gs://neulab-qa/t5-data/pretrained_models/3B
#from_model_dir=gs://neulab-qa/t5-data/pretrained_models/3B/model.ckpt-1000000
#to_model_dir=gs://neulab-qa/t5-data/pretrained_models/3B_ft
from_model=gs://neulab-qa/unifiedqa/models/3B/model.ckpt-1100500
to_model_dir=gs://neulab-qa/unifiedqa/models/${output}
model_parallelism=8
train_steps=1101000
tpb=16384

./run_test.py \
    --tpu="${tpu_name}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${to_model_dir}" \
    --gin_file="${gin_model_dir}/operative_config.gin" \
    --t5_tfds_data_dir="${DATA_DIR}" \
    --gin_file="dataset.gin" \
    --gin_param="run.init_checkpoint = '${from_model}'" \
    --gin_param="utils.tpu_mesh_shape.model_parallelism = ${model_parallelism}" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
    --gin_param="MIXTURE_NAME = 'uq_train_mix'" \
    --gin_param="run.train_steps = ${train_steps}" \
    --gin_param="build_uq.neg_method = '${neg_method}'" \
    --gin_param="run.batch_size = ('tokens_per_batch', ${tpb})"
