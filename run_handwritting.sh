#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES=0

python run_LLM4TS.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./dataset/Handwriting/ \
    --model_id LLM4TS_cls_Handwriting \
    --model LLM4TS_cls \
    --data UEA \
    --is_llm 1 \
    --pretrain 1 \
    --freeze 1 \
    --llm_layers 6 \
    --llm openai-community/gpt2 \
    --d_model 768 \
    --d_ff 768 \
    --patch_len 16 \
    --stride 16 \
    --train_epochs 100 \
    --patience 10 \
    --itr 1 \
    --batch_size 16 \
    --learning_rate 0.002 \
    --random_seed 2021 \
    --pt_sft 1 \
    --pt_sft_base_dir ./checkpoints/pt_patch/ETTh1_ETTm1_ETTh2_ETTm2_weather_traffic_electricity_illness/ \
    --pt_sft_model pretrain_LLM4TS_pt_sl1024_pl1024_llml6_lr0.0001_bs256_percent100_ln_wpe_attn_mlp_gpt2_w_weight_s16_0 \
    --sft 1 \
    --sft_layers ln_wpe \
    --checkpoints ./checkpoints/classification/Handwriting \
    --des exp \
    --lradj type1 \
    --use_gpu 1 \
    --devices 0 \
    --gpu 0 \
    --num_workers 0 2>&1 | tee logs/classification/Handwriting.log