#!/usr/bin/env bash

tpu_name=$1  # default-dgdw2, jzb

# model parameters
output=$2
model_type=$3  # 3B, 11B
model_dir=$4  # unifiedqa/models/3B, t5-data/pretrained_models/3B
model_dir=gs://neulab-qa/${model_dir}
step=$5  # 1100500

# dataset parameters
mix=$6
split=$7
neg_method=weight
ret_method='q-append'
ret_ind=0

model_gin_file=t5_${model_type}_operative_config.gin
model_parallelism=8
inp_len=512
tgt_len=128
if [[ $model_type == '3B' ]]; then
    tpb=262144
elif [[ $model_type == '11B' ]]; then
    tpb=65536
fi

mkdir -p $(dirname "${output}")

./run_test.py \
    --tpu="${tpu_name}"  \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${model_dir}" \
    --gin_file="${model_gin_file}" \
    --t5_tfds_data_dir="${DATA_DIR}" \
    --gin_file="score_from_task.gin" \
    --gin_file="greedy_decode.gin" \
    --gin_param="MIXTURE_NAME = '${mix}'" \
    --gin_param="run.dataset_split = '${split}'" \
    --gin_param="score_from_dataset.scores_filename = '${output}'" \
    --gin_param="utils.tpu_mesh_shape.model_parallelism = ${model_parallelism}" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
    --gin_param="utils.run.eval_checkpoint_step = ${step}" \
    --gin_param="utils.run.sequence_length = {'inputs': ${inp_len}, 'targets': ${tgt_len}}" \
    --gin_param="mesh_eval_dataset_fn.num_eval_examples = None" \
    --gin_param="build.neg_method = '${neg_method}'" \
    --gin_param="build.ret_method = '${ret_method}'" \
    --gin_param="build.ret_ind = ${ret_ind}" \
    --gin_param="run.batch_size = ('tokens_per_batch', ${tpb})"
