import os
import torch
import numpy as np
import pickle
import sys
sys.path.append('../')
from utils import get_activations_bau, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q
from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions, svd_decomposition,svd_decomposition_2
import llama
import qwen2
import argparse
import json
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str,default="Qwen1.5-14B-Chat")
parser.add_argument('--main_steer_style', type=str,default="DRC",help="主要采用的数据集的风格")
parser.add_argument('--second_steer_style', type=str,default="None",help="次要采用的数据集的风格")
parser.add_argument('--dataset_name', type=str, default="DRC", help='测试用的数据集')
parser.add_argument('--baseline_name', type=str, default="caa", help="caa, lm_steer, vector_prompt, iti, None, ablation")
parser.add_argument('--main_strength', type=float, default=3.0, help='main steer strength')
parser.add_argument('--second_strength', type=float, default=3.0, help='second steer strength')
parser.add_argument('--is_heads', type=int, default=0, help='分析实验 命名')
parser.add_argument('--num_heads', type=int, default=64, help='分析实验 命名')
parser.add_argument('--K', type=int, default=None, help='奇异值')
args = parser.parse_args()

"""
{
        "input": "如果你走进一个衣柜，在现实生活中会发生什么？",
        "orig_pred": [
            "如果你走进一个衣柜，在现实生活中，你可能会发现自己卡在了狭小的空间里，周围是衣物。其他人可能会进来寻找你，或者以为你在玩捉迷藏。如果衣柜设计得足够大，可能是一个密室逃脱游戏的一部分，但通常这不是日常生活中会遇到的情况。"
        ],
        "pred": [
            "你一头撞在门框上，定睛一看，原来自己站错了地儿，这是做梦呢！真走进去，可不闷死个人！"
        ],
        "reference_response": null,
        "complete_output": [
            "<|im_start|>system\n请你以对话的形式直接对下面的语句作出回应：<|im_end|>\n<|im_start|>user\n如果你走进一个衣柜，在现实生活中会发生什么？<|im_end|>\n<|im_start|>assistant\n你一头撞在门框上，定睛一看，原来自己站错了地儿，这是做梦呢！真走进去，可不闷死个人！<|im_end|>"
        ]
    },
"""

if args.baseline_name == "None":
    file_path = f"results_log/{args.model_name}_eval{args.dataset_name}_main{args.main_steer_style}_second{args.second_steer_style}_strength{args.main_strength}_{args.second_strength}_result.json"
elif args.baseline_name in ["caa", "lm_steer", "vector_prompt", "iti","prompt"]:
    file_path = f"results_log/baseline_result/{args.model_name}/{args.main_steer_style}/{args.baseline_name}/{args.dataset_name}_results.json"  
elif "ablation_svd" in args.baseline_name:
    file_path = f"results_log/{args.model_name}_eval{args.dataset_name}_main{args.main_steer_style}_second{args.second_steer_style}_strength{args.main_strength}_{args.second_strength}_result_{args.baseline_name}.json"
elif  "ablation_fixed" in args.baseline_name:
    file_path = f"results_log/{args.model_name}_eval{args.dataset_name}_main{args.main_steer_style}_second{args.second_steer_style}_strength{args.main_strength}_{args.second_strength}_result_{args.baseline_name}.json"
elif args.is_heads==1:
    file_path = f"results_log/{args.model_name}_eval{args.dataset_name}_main{args.main_steer_style}_second{args.second_steer_style}_strength{args.main_strength}_{args.second_strength}_head{args.num_heads}_result.json"
elif args.K != None:
    file_path = f"results_log/{args.model_name}_eval{args.dataset_name}_main{args.main_steer_style}_second{args.second_steer_style}_strength{args.main_strength}_{args.second_strength}_K{args.K}_result.json"

with open(file_path, 'r', encoding='utf-8') as f:
    data_list = json.load(f)


# import ipdb; ipdb.set_trace()
main_steer_style = args.main_steer_style # 采用超参数风格
origin_answer_path = f"results_log/{args.model_name}_eval{args.dataset_name}_origin_result.json"
with open(origin_answer_path, 'r', encoding='utf-8') as f:
    ori_data_list = json.load(f)

# import ipdb; ipdb.set_trace()
# ===================== 数据划分 ====================
# 读取基于steer_style计算并保存好的steer向量

def main():
        
    # q_tokens = tokenizer(question, return_tensors = 'pt').input_ids
    # get_activations(q_tokens)
    output_data = []
    # import  ipdb; ipdb.set_trace()
    if args.baseline_name == "None" :
        for i in range(len(data_list)):
            dict = {}
            dict["question"] = data_list[i]["question"]
            dict["model_completion"] = data_list[i]["model_completion"]
            dict["origin_model_completion"] = ori_data_list[i]["origin_model_completion"]
            # dict["model_path"] = model_path
            output_data.append(dict)
    elif args.baseline_name in ["caa", "lm_steer", "vector_prompt"]:
        for i in range(len(data_list)):
            dict = {}
            dict["question"] = data_list[i]["input"]
            dict["origin_model_completion"] = ori_data_list[i]["origin_model_completion"]
            dict["model_completion"] = data_list[i]["pred"][0]
            output_data.append(dict)
        # for i in range(len(data_list)):# 临时
        #     dict = {}
        #     dict["question"] = data_list[i]["question"]
        #     dict["model_completion"] = data_list[i]["model_completion"][0]
        #     dict["origin_model_completion"] = ori_data_list[i]["origin_model_completion"]
        #     # dict["model_path"] = model_path
        #     output_data.append(dict)
    elif args.baseline_name in ['iti','prompt']:
        for i in range(len(data_list)):
            dict = {}
            dict["question"] = data_list[i]["question"]
            dict["model_completion"] = data_list[i]["model_completion"]
            dict["origin_model_completion"] = ori_data_list[i]["origin_model_completion"]
            # dict["model_path"] = model_path
            output_data.append(dict)

    elif "ablation" in args.baseline_name:
        for i in range(len(data_list)):
            # import ipdb; ipdb.set_trace()
            dict = {}
            dict["question"] = data_list[i]["question"]
            dict["model_completion"] = data_list[i]["model_completion"]
            dict["origin_model_completion"] = ori_data_list[i]["origin_model_completion"]
            output_data.append(dict)
    else:
        raise ValueError("Invalid baseline name")
    # ============== 存储 ============
    with open(file_path, 'w', encoding='utf-8') as new_file:
        json.dump(output_data, new_file, ensure_ascii=False, indent=4)
    print(f"Results saved to {file_path}")
#==================================================================================================
if __name__ == "__main__":
    main()

