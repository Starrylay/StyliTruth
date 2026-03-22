# python edit_weight.py --model_name Qwen1.5-14B-Chat --dataset_name DRC --activation_path "" --label_path "" --model_dir "/data/CharacterAI/PretainedModels/Qwen1.5-14B-Chat" --num_heads 64 --alpha 3
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import pickle
import os
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import sys
sys.path.append('../')
from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions
# import llama
import qwen2
import llama
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='Qwen1.5-14B-Chat', help='model name')
    parser.add_argument("--dataset_name", type=str, default=None, help='dataset name')
    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    # 以上为必需参数
    parser.add_argument('--num_heads', type=int, default=64, help='K, number of top heads to intervene on')
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--iti_middle_save', type=int, default=0)
    parser.add_argument('--token_position', type=str, default='last', help='token position to extract activations')
    args = parser.parse_args()

    # args.num_heads= 2
    print("top num_heads", args.num_heads)

    # set seeds
    print("set seeds")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create model

    if args.model_name == "Qwen1.5-14B-Chat":
        MODEL = "your own model path/Qwen1.5-14B-Chat"
    

    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    model = qwen2.Qwen2ForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")

    # define number of layers and heads
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    # load activations 

    activation_path = f"features/{args.dataset_name}/{args.model_name}_{args.dataset_name}_{args.token_position}_head_wise.npy"
    print("load activations from ", activation_path)
    label_path = f'features/{args.dataset_name}/{args.model_name}_{args.dataset_name}_{args.token_position}_labels.npy'
    print("load labels from ", label_path)

    head_wise_activations = np.load(f"{activation_path}")
    labels = np.load(f"{label_path}")
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)
    print(head_wise_activations.shape)

    dataset_len = head_wise_activations.shape[0] // 2

    # tuning dataset: no labels used, just to get std of activations along the direction
    tuning_activations = np.load(f"{activation_path}")
    tuning_activations = rearrange(tuning_activations, 'b l (h d) -> b l h d', h = num_heads)
    tuning_labels = np.load(f"{label_path}")


    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)
    train_idxs = np.arange(dataset_len)

    # pick a val set using numpy
    train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
    val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

    com_directions = None
    top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir,args.dataset_name)
    print("top heads:", len(top_heads))
    
    np.save(f"features/{args.dataset_name}/{args.model_name}_probes_{args.num_heads}_{args.token_position}.npy",probes)
    np.save(f"features/{args.dataset_name}/{args.model_name}_top_heads_{args.num_heads}_{args.token_position}.npy",top_heads)
    interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads, args.use_center_of_mass, args.use_random_dir, com_directions)



    activations_dict = {} # save
    for head_out_name, list_int_vec in tqdm(interventions.items()):
        layer_no = int(head_out_name.split('.')[2])
        displacement = np.zeros((int(num_heads), int(hidden_size / num_heads)))
        activations_dict[layer_no] = {} # save
        for head_no, head_vec, std in list_int_vec: # 此处 head_no 是 head index
            activations = tuning_activations[:,layer_no,head_no,:]
            correct_activations = activations[::2, :]
            incorrect_activations = activations[1::2, :]
            correct_activations = np.mean(correct_activations, axis=0)
            incorrect_activations = np.mean(incorrect_activations, axis=0)
            # displacement[head_no] = args.alpha * (correct_activations - incorrect_activations)
            displacement[head_no] = correct_activations - incorrect_activations
            activations_dict[layer_no][head_no] = displacement[head_no] # save  只有key 会在后边用到，其他的都没用
            # print(layer_no,head_no)

    with open(f"features/{args.dataset_name}/{args.model_name}_activations_{args.num_heads}_{args.token_position}.pkl", 'wb') as f:
        pickle.dump(activations_dict, f)
    print("save results")
        
if __name__ == "__main__":
    main()
