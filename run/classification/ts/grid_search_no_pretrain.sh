#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES=0

echo "Running grid search for patch and stride on EthanolConcentration dataset"
./run_ethanol_no_pretrain.sh

echo "Running grid search for patch and stride on FaceDetection dataset"
./run_face_detection_no_pretrain.sh

echo "Running grid search for patch and stride on Handwriting dataset"
./run_handwriting_no_pretrain.sh

echo "Running grid search for patch and stride on Heartbeat dataset"
./run_heartbeat_no_pretrain.sh

echo "Running grid search for patch and stride on JapaneseVowels dataset"
./run_japanese_vowels_no_pretrain.sh

echo "Running grid search for patch and stride on PEMS-SF dataset"
./run_pems_sf_no_pretrain.sh

echo "Running grid search for patch and stride on SelfRegulationSCP1 dataset"
./run_self_regulation_scp1_no_pretrain.sh

echo "Running grid search for patch and stride on SelfRegulationSCP2 dataset"
./run_self_regulation_scp2_no_pretrain.sh

echo "Running grid search for patch and stride on SpokenArabicDigits dataset"
./run_spoken_arabic_digits_no_pretrain.sh

echo "Running grid search for patch and stride on UWaveGestureLibrary dataset"
./run_uwave_gesture_library_no_pretrain.sh