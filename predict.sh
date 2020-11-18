#!/usr/bin/env bash

tpu_name=$1  # default-dgdw2, jzb

# model parameter
model_type=$2  # 3B, 11B
model_dir=gs://neulab-qa/unifiedqa/models/${model_type}
step=1100500
output=$3

# dataset parameter
mix=$4
split=$5
search_alg=sample_decode
beam_size=1
temp=2.5
max_decode_length=128

if [[ $model_type == '3B' ]]; then
    tpb=131072
elif [[ $model_type == '11B' ]]; then
    tpb=65536
fi
inp_len=512
tgt_len=128
gin_model_dir=gs://neulab-qa/t5-data/pretrained_models/${model_type}
model_parallelism=8

mkdir -p $(dirname "${output}")

./run_test.py \
    --tpu=${tpu_name} \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${model_dir}" \
    --gin_file="${gin_model_dir}/operative_config.gin" \
    --t5_tfds_data_dir="${DATA_DIR}" \
    --gin_file="infer.gin" \
    --gin_file="${search_alg}.gin" \
    --gin_param="mesh_eval_dataset_fn.mixture_or_task_name = '${mix}'" \
    --gin_param="mesh_eval_dataset_fn.num_eval_examples = None" \
    --gin_param="run.dataset_split = '${split}'" \
    --gin_param="output_filename = '${output}'" \
    --gin_param="utils.run.sequence_length = {'inputs': ${inp_len}, 'targets': ${tgt_len}}" \
    --gin_param="utils.tpu_mesh_shape.model_parallelism = ${model_parallelism}" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
    --gin_param="infer_checkpoint_step = ${step}" \
    --gin_param="run.batch_size = ('tokens_per_batch', ${tpb})" \
    --gin_param="Bitransformer.decode.max_decode_length = ${max_decode_length}" \
    --gin_param="Bitransformer.decode.beam_size = ${beam_size}" \
    --gin_param="Bitransformer.decode.temperature = ${temp}"
