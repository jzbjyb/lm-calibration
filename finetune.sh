#!/usr/bin/env bash

neg_method=$1
output=$2
train_steps=$3  # 1105000
loss=$4

tpu_name=jzb	# default-dgdw2
mix=uq_ext_decode_train_ol_ans_mix
split=dev
num_sep=5
tgt_len=2560
gin_model_dir=gs://neulab-qa/t5-data/pretrained_models/3B
#from_model_dir=gs://neulab-qa/t5-data/pretrained_models/small/model.ckpt-1000000
#to_model_dir=gs://neulab-qa/t5-data/pretrained_models/small_ft
from_model=gs://neulab-qa/unifiedqa/models/3B/model.ckpt-1100500
to_model_dir=gs://neulab-qa/unifiedqa/models/${output}
model_parallelism=8
tpb=15360  # 16384

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
    --gin_param="MIXTURE_NAME = '${mix}'" \
    --gin_param="mesh_train_dataset_fn.dataset_split = '${split}'" \
    --gin_param="run.train_steps = ${train_steps}" \
    --gin_param="build.neg_method = '${neg_method}'" \
    --gin_param="run.batch_size = ('tokens_per_batch', ${tpb})" \
    --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': ${tgt_len}}" \
    --gin_param="encoder/Unitransformer.z_loss = 0.0" \
    --gin_param="decoder/Unitransformer.z_loss = 0.0" \
    --gin_param="Bitransformer.num_sep = ${num_sep}" \
    --gin_param="Bitransformer.loss_type = '${loss}'"
