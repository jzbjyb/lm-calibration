#!/usr/bin/env bash

neg_method=$1
output=$2
model_dir=$3  # gs://neulab-qa/unifiedqa/models/3B, gs://neulab-qa/t5-data/pretrained_models/3B
step=$4  # 1100500
mix=$5

tpu_name=jzb
gin_model_dir=gs://neulab-qa/t5-data/pretrained_models/3B
tpb=32768
split=dev

./run_test.py \
    --tpu="${tpu_name}"  \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${model_dir}" \
    --gin_file="${gin_model_dir}/operative_config.gin" \
    --t5_tfds_data_dir="${DATA_DIR}" \
    --gin_file="score_from_task.gin" \
    --gin_file="greedy_decode.gin" \
    --gin_param="MIXTURE_NAME = '${mix}'" \
    --gin_param="run.dataset_split = '${split}'" \
    --gin_param="score_from_dataset.scores_filename = '${output}'" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
    --gin_param="utils.run.eval_checkpoint_step = ${step}" \
    --gin_param="mesh_eval_dataset_fn.num_eval_examples = None" \
    --gin_param="build_uq.neg_method = '${neg_method}'" \
    --gin_param="run.batch_size = ('tokens_per_batch', ${tpb})"
