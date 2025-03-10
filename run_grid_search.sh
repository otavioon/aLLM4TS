#!/bin/bash

set -x

export CUDA_VISIBLE_DEVICES=0

echo "Running grid search for patch and stride on EthanolConcentration dataset"
./run_ethanol.sh

echo "Running grid search for patch and stride on FaceDetection dataset"
./run_face_detection.sh

echo "Running grid search for patch and stride on Handwriting dataset"
./run_handwriting.sh

echo "Running grid search for patch and stride on Heartbeat dataset"
./run_heartbeat.sh

echo "Running grid search for patch and stride on JapaneseVowels dataset"
./run_japanese_vowels.sh

echo "Running grid search for patch and stride on PEMS-SF dataset"
./run_pems_sf.sh

echo "Running grid search for patch and stride on SelfRegulationSCP1 dataset"
./run_self_regulation_scp1.sh

echo "Running grid search for patch and stride on SelfRegulationSCP2 dataset"
./run_self_regulation_scp2.sh

echo "Running grid search for patch and stride on SpokenArabicDigits dataset"
./run_spoken_arabic_digits.sh

echo "Running grid search for patch and stride on UWaveGestureLibrary dataset"
./run_uwave_gesture_library.sh