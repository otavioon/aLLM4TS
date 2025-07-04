#!/bin/bash


set -x
cd ../../..

valid_datasets=("KuHar" "MotionSense" "RealWorld_thigh" "RealWorld_waist" "UCI" "WISDM")

if [[ ! " ${valid_datasets[@]} " =~ " $1 " ]]; then
    echo "Error: Invalid dataset. Valid options are: KuHar, MotionSense, RealWorld_thigh, RealWorld_waist, UCI, WISDM"
    exit 1
fi

dataset=$1
pretrain_codes=("A" "AE" "AD")
pretrain_dsets_suffix=("" "-ES" "-DAGHAR")

head="allm4ts"
# head="hiaac"

mkdir -p logs/classification_daghar_no_pretrain

# for patch in 8 16 ; do
    # for stride in 16 8 4 2 ; do
        # for instance_norm in "yes" "no"; do

for i in 0; do  # only A (does not have any pretrain...)
    for instance_norm in "yes"; do
        for patch in 8 ; do
            for stride in  8 ; do
                for percent in 1 10 50 100 ; do
                    # for freeze in 0 1 2 ; do
                    for freeze in 0 2 ; do
                        pt_code=${pretrain_codes[i]}
                        pt_dset_suffix=${pretrain_dsets_suffix[i]}
                        python run_LLM4TS.py \
                            --task_name classification \
                            --is_training 1 \
                            --root_path /workspaces/HIAAC-KR-Dev-Container/some_datasets/DAGHAR/standardized_view/${dataset}/ \
                            --model_id LLM4TS_cls_${dataset} \
                            --model LLM4TS_cls \
                            --data DAGHAR \
                            --is_llm 1 \
                            --pretrain 1 \
                            --freeze ${freeze} \
                            --llm_layers 6 \
                            --llm openai-community/gpt2 \
                            --d_model 768 \
                            --d_ff 768 \
                            --patch_len ${patch} \
                            --stride ${stride} \
                            --train_epochs 100 \
                            --patience 10 \
                            --itr 3 \
                            --batch_size 128 \
                            --learning_rate 0.002 \
                            --random_seed 2021 \
                            --pt_sft 0 \
                            --pt_sft_base_dir ./checkpoints/experiments/aLLM4TS-${pt_code}/ETTh1_ETTm1_ETTh2_ETTm2_weather_traffic_electricity_illness${pt_dset_suffix} \
                            --pt_sft_model pretrain_LLM4TS_pt_sl1024_pl1024_llml6_lr0.0001_bs256_percent100_ln_wpe_attn_mlp_gpt2_w_weight_s16_0 \
                            --sft 1 \
                            --sft_layers ln_wpe_attn_mlp \
                            --checkpoints ./checkpoints/classification_daghar_no_pretrain/${dataset}_patch-${patch}_stride-${stride}_aLLM4TS-${pt_code}_norm-${instance_norm}_percent-${percent}_freeze-${freeze}_head-${head} \
                            --des exp \
                            --lradj type1 \
                            --use_gpu 1 \
                            --devices 0 \
                            --gpu 0 \
                            --perform_instance_norm ${perform_instance_norm} \
                            --percent ${percent} \
                            --head ${head} \
                            --num_workers 0 2>&1 | tee logs/classification_daghar_no_pretrain/${dataset}_patch-${patch}_stride-${stride}_aLLM4TS-${pt_code}_norm-${instance_norm}_percent-${percent}_freeze-${freeze}_head-${head}.log
                    done
                done
            done
        done
    done
done