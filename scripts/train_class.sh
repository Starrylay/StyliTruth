#!/bin/bash

# export CUDA_VISIBLE_DEVICES=2

# #=========================tqa_gen_zh==============================
LOG_FILE="logs/eval_train_$(date +%Y%m%d_%H%M%S).log"
echo "===== eval train =====" | tee -a $LOG_FILE

nohup python   evaluation/train_classifier.py \
>> $LOG_FILE 2>&1