#!/bin/bash

export CUDA_VISIBLE_DEVICES=3


MAIN_STRENGTHS=(0.0 0.5 1.0 1.5 2.0)
SECOND_STRENGTHS=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)
HEAD=(32)
MODELs=("Qwen1.5-14B-Chat")
Token_Position="last"
for MODEL in "${MODELs[@]}"; do
  for head in "${HEAD[@]}"; do
    LOG_FILE="logs/run_rebuttal_other_model_${MODEL}_$(date +%Y%m%d_%H%M%S).log"
    echo "Step 1: step1_get_activations ..." | tee -a $LOG_FILE
    nohup python step1_get_activations.py \
        --model_name "$MODEL" \
        --dataset_name DRC \
        --debug 0\
        --token_position "$Token_Position"\
        >> $LOG_FILE 2>&1
    wait

    echo "Step 1: step1_get_activations..." | tee -a $LOG_FILE
    nohup python step1_get_activations.py \
        --model_name "$MODEL" \
        --dataset_name tqa_gen_zh_all \
        --debug 0\
        --token_position "$Token_Position"\
        >> $LOG_FILE 2>&1
    wait
    
    echo "Step 2: Editing model weights ..." | tee -a $LOG_FILE
    nohup python step2_edit_weight.py \
        --model_name "$MODEL" \
        --dataset_name DRC \
        --num_heads "$head"\
        --token_position "$Token_Position"\
        >> $LOG_FILE 2>&1
    wait

    echo "Step 2: Editing model weights ..." | tee -a $LOG_FILE
    nohup python step2_edit_weight.py \
        --model_name "$MODEL" \
        --dataset_name tqa_gen_zh_all \
        --num_heads "$head"\
        --token_position "$Token_Position"\
        >> $LOG_FILE 2>&1
    wait

    for main_s in "${MAIN_STRENGTHS[@]}"; do
      for second_s in "${SECOND_STRENGTHS[@]}"; do
          LOG_FILE="logs/run_rebuttal_other_model_${MODEL}_ms${main_s}_ss${second_s}_$(date +%Y%m%d_%H%M%S).log"
          echo "===== Running with MODEL=${MODEL}, HEAD=${head},ms=${main_s},ss=${second_s} =====" | tee -a "$LOG_FILE"
          echo "Step 3.1: Generating answers with edited model..." | tee -a "$LOG_FILE"
          nohup python step3_generate.py \
            --model_name "$MODEL" \
            --main_steer_style DRC \
            --second_steer_style tqa_gen_zh_all \
            --dataset_name tqa_gen_zh \
            --num_heads "$head" \
            --debug 0 \
            --main_strength "$main_s" \
            --second_strength "$second_s" \
            --is_heads 0 \
            --token_position "$Token_Position"\
          >> "$LOG_FILE" 2>&1 &
          wait
          nohup python step3.2_only_run_one_generate_origin.py \
            --model_name "$MODEL"\
            --main_steer_style None\
            --second_steer_style None\
            --dataset_name tqa_gen_zh\
            --num_heads "$head"  \
            --debug 0\
          >> "$LOG_FILE" 2>&1 &

          echo "Step 3.5.3: add_origin..." | tee -a "$LOG_FILE"
          nohup python step3.5_add_origin.py \
            --model_name "$MODEL" \
            --main_steer_style DRC \
            --second_steer_style tqa_gen_zh_all \
            --dataset_name tqa_gen_zh \
            --baseline_name None \
            --main_strength "$main_s" \
            --second_strength "$second_s" \
            --is_heads 0 \
            --num_heads "$head" \
          >> "$LOG_FILE" 2>&1 &
          wait

          echo "step4.3: evaluation" | tee -a "$LOG_FILE"
          nohup python evaluation/evaluation.py \
            --model_name "$MODEL" \
            --main_steer_style DRC \
            --second_steer_style tqa_gen_zh_all \
            --dataset_name tqa_gen_zh \
            --baseline_name None \
            --main_strength "$main_s" \
            --second_strength "$second_s" \
            --is_heads 0 \
            --num_heads "$head" \
            >> "$LOG_FILE" 2>&1 &
          wait
    
          echo "step5.3: evaluation tqa" | tee -a "$LOG_FILE"
          nohup python step5_tfqa_eval.py \
            --model_name "$MODEL" \
            --main_steer_style DRC \
            --second_steer_style tqa_gen_zh_all \
            --dataset_name tqa_gen_zh \
            --baseline_name None \
            --main_strength "$main_s" \
            --second_strength "$second_s" \
            --is_heads 0 \
            --num_heads "$head" \
            --debug 0 \
          >> "$LOG_FILE" 2>&1 &
          wait

          # echo "step6: all score" | tee -a "$LOG_FILE"
          nohup python step6_score_all.py \
            --model_name "$MODEL" \
            --main_steer_style DRC \
            --second_steer_style tqa_gen_zh_all \
            --dataset_name tqa_gen_zh \
            --baseline_name None \
            --main_strength "$main_s" \
            --second_strength "$second_s" \
            --is_heads 0 \
            --num_heads "$head" \
            --debug 0 \
          >> "$LOG_FILE" 2>&1 &
          wait
      done
    done
  done
done

