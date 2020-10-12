#!/usr/bin/env bash

model_dir=$1  # gs://neulab-qa/unifiedqa/models/3B, gs://neulab-qa/t5-data/pretrained_models/3B
step=$2  # 1100500
output=$3
mix=$4
split=$5

tpu_name=jzb  # default-dgdw2
gin_model_dir=gs://neulab-qa/t5-data/pretrained_models/3B
model_parallelism=8
tpb=32768

mkdir -p $(dirname "${output}")

./run_test.py \
    --tpu=${tpu_name} \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${model_dir}" \
    --gin_file="${gin_model_dir}/operative_config.gin" \
    --t5_tfds_data_dir="${DATA_DIR}" \
    --gin_file="infer.gin" \
    --gin_file="beam_search.gin" \
    --gin_param="mesh_eval_dataset_fn.mixture_or_task_name = '${mix}'" \
    --gin_param="mesh_eval_dataset_fn.num_eval_examples = None" \
    --gin_param="run.dataset_split = '${split}'" \
    --gin_param="output_filename = '${output}'" \
    --gin_param="utils.tpu_mesh_shape.model_parallelism = ${model_parallelism}" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
    --gin_param="infer_checkpoint_step = ${step}" \
    --gin_param="run.batch_size = ('tokens_per_batch', ${tpb})" \
    --gin_param="Bitransformer.decode.max_decode_length = 128" \
    --gin_param="Bitransformer.decode.beam_size = 5"
