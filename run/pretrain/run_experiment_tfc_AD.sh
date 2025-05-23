#!/bin/bash

cd ../..

export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="./checkpoints/experiments/TFC-AD/ETTh1_ETTm1_ETTh2_ETTm2_weather_traffic_electricity_illness-DAGHAR/"
MODEL_NAME="tfc-transformer-encoder"
CKPT_VERSION="final"
FINAL_DIR=${CKPT_DIR}/${MODEL_NAME}/${CKPT_VERSION}

mkdir -p ${FINAL_DIR}


python run_tfc_pretrain.py   \
    --checkpoints ${CKPT_DIR} \
    --model_name ${MODEL_NAME} \
    --ckpt_version ${CKPT_VERSION} \
    --input_channels 1 \
    --TS_length 1024 \
    --batch_size 128 \
    --num_workers 0 \
    --data pretrain_allm4ts_daghar \
    --train_epochs 100 \
    --accelerator gpu \
    --devices 1 \
    --resume_from last.ckpt 2>&1 | tee ${CKPT_DIR}/log.txt