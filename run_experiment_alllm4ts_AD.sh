#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

CKPT_DIR=./checkpoints/experiments/aLLM4TS-AD/ETTh1_ETTm1_ETTh2_ETTm2_weather_traffic_electricity_illness-DAGHAR
mkdir -p ${CKPT_DIR}

python run_LLM4TS.py \
--is_training 1 \
--root_path ./dataset/ \
--data_path null \
--model_id pretrain_LLM4TS_pt \
--model LLM4TS_pt \
--data pretrain_allm4ts_daghar \
--percent 100 \
--features M \
--seq_len 1024 \
--label_len 0 \
--pred_len 1024 \
--is_llm 1 \
--task_name 1 \
--freeze 1 \
--llm_layers 6 \
--llm openai-community/gpt2 \
--affine 1 \
--enc_in 1 \
--e_layers 4 \
--n_heads 4 \
--d_model 768 \
--d_ff 768 \
--dropout 0.2 \
--fc_dropout 0.2 \
--head_dropout 0 \
--patch_len 16 \
--stride 16 \
--des ln_wpe_attn_mlp_gpt2_w_weight_s16 \
--train_epochs 100 \
--patience 5 \
--itr 1 \
--batch_size 256 \
--learning_rate 0.0001 \
--c_pt 1 \
--pt_data ETTh1_ETTm1_ETTh2_ETTm2_weather_traffic_electricity_illness \
--pt_layers ln_wpe_attn_mlp \
--checkpoints ${CKPT_DIR} \
--use_gpu 1 \
--devices 0 \
--gpu 0 \
--num_workers 0 2>&1 | tee ${CKPT_DIR}/log.txt


# Num Workers = 0 --> https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/17
