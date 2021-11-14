#!/usr/bin/env bash

tpu_name=$1	# default-dgdw2, jzb

# model parameters
from_model=$2  # 3B, 11B
to_model=$3
loss=$4
neg_method=weight

# dataset parameters
mix=$5
split=$6
format=$7

if [[ $format == 'mc' ]]; then
    num_sep=8
    inp_len=512
    tgt_len=1024  # 128 * 8
    if [[ $from_model == '3B' ]]; then
        tpb=16384  # 128 * 8 * 16
        train_steps=1105000
    elif [[ $from_model == '11B' ]]; then
        tpb=3072  # 128 * 8 * 3
        train_steps=1125000
    fi
elif [[ $format == 'ext' ]]; then
    num_sep=5
    inp_len=512
    tgt_len=640  # 128 * 5
    if [[ $from_model == '3B' ]]; then
        tpb=20480  # 128 * 5 * 32
        train_steps=1105000
    elif [[ $from_model == '11B' ]]; then
        tpb=3200  # 128 * 5 * 5
        train_steps=1125000
    fi
elif [[ $format == 'mh' ]]; then
    num_sep=1
    inp_len=512
    tgt_len=128  # 128 * 1
    if [[ $from_model == '3B' ]]; then
        tpb=16384  # 128 * 1 * 128
        train_steps=1105000
    elif [[ $from_model == '11B' ]]; then
        tpb=2048  # 128 * 1 * 16
        train_steps=1200000
    fi
fi

echo $tpu_name $from_model $to_model $loss $mix $split $format
echo 'num sep' $num_sep 'inp len' $inp_len 'tgt len' $tgt_len 'token per batch' $tpb 'step' $train_steps

model_gin_file=t5_${from_model}_operative_config.gin
from_model=gs://neulab-qa/unifiedqa/models/${from_model}/model.ckpt-1100500
to_model_dir=gs://neulab-qa/unifiedqa/ft_models/${to_model}
model_parallelism=8

./run_test.py \
    --tpu="${tpu_name}" \
    --gcp_project="${PROJECT}" \
    --tpu_zone="${ZONE}" \
    --model_dir="${to_model_dir}" \
    --gin_file="${model_gin_file}" \
    --t5_tfds_data_dir="${DATA_DIR}" \
    --gin_file="dataset.gin" \
    --gin_file="learning_rate_schedules/constant_0_001.gin" \
    --gin_param="run.init_checkpoint = '${from_model}'" \
    --gin_param="utils.tpu_mesh_shape.model_parallelism = ${model_parallelism}" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
    --gin_param="MIXTURE_NAME = '${mix}'" \
    --gin_param="mesh_train_dataset_fn.dataset_split = '${split}'" \
    --gin_param="run.train_steps = ${train_steps}" \
    --gin_param="build.neg_method = '${neg_method}'" \
    --gin_param="run.batch_size = ('tokens_per_batch', ${tpb})" \
    --gin_param="utils.run.sequence_length = {'inputs': ${inp_len}, 'targets': ${tgt_len}}" \
    --gin_param="encoder/Unitransformer.z_loss = 0.0" \
    --gin_param="decoder/Unitransformer.z_loss = 0.0" \
    --gin_param="Bitransformer.num_sep = ${num_sep}" \
    --gin_param="Bitransformer.loss_type = '${loss}'"
