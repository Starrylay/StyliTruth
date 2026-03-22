# python get_activations.py Qwen1.5-14B-Chat DRC --model_dir "/data/CharacterAI/PretainedModels/Qwen1.5-14B-Chat"
import os
import torch
import numpy as np
import pickle
from utils import get_activations_bau, tokenized_tqa, tokenized_tqa_gen_DRC, tokenized_tqa_gen_Shakespeare,tokenized_tqa_gen,tokenized_tqa_gen_zh,tokenized_tqa_gen_zh_all,tokenized_tqa_gen_all
import qwen2
import llama
import argparse
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen1.5-14B-Chat')
    parser.add_argument('--dataset_name', type=str, default='Daiyu')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--debug", type=int, default=1, help='if set, only use 100 samples for debugging')
    parser.add_argument('--token_position', type=str, default='last', help='which token to get activations for: last or all')
    args = parser.parse_args()


    if args.model_name == "Qwen1.5-14B-Chat":
        MODEL = "your own model path/Qwen1.5-14B-Chat"
    
    
    tokenizer = qwen2.Qwen2Tokenizer.from_pretrained(MODEL)
    model = qwen2.Qwen2ForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")

    
    
    device = "cuda"
    if args.dataset_name == "DRC": 
        with open("dataset/Train_DRC.json", 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        formatter = tokenized_tqa_gen_DRC

    elif args.dataset_name == "tqa_gen_zh": 
        with open("dataset/Train_tqa_gen_zh.json", 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        formatter = tokenized_tqa_gen_zh
    elif args.dataset_name == "tqa_gen_zh_all": 
        with open("dataset/Train_tqa_gen_zh.json", 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        formatter = tokenized_tqa_gen_zh_all
    elif args.dataset_name == "Shakespeare":
        with open("dataset/Train_Shakespeare.json", 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        formatter = tokenized_tqa_gen_Shakespeare
    elif args.dataset_name == "tqa_gen": 
        with open("dataset/Train_tqa_gen.json", 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        formatter = tokenized_tqa_gen  
    elif args.dataset_name == "tqa_gen_all": 
        with open("dataset/Train_tqa_gen.json", 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        formatter = tokenized_tqa_gen_all  
    else: 
        raise ValueError("Invalid dataset name")
    if args.debug==1:
        # 随机采样1000条数据
        import random
        random.seed(42)
        dataset = random.sample(dataset, 50)

    print("Tokenizing prompts")
    print(len(dataset))
    prompts, labels = formatter(dataset, tokenizer)
    print(len(prompts), len(labels))
    all_layer_wise_activations = []
    all_head_wise_activations = []
    print("Getting activations")

    if "instruct" in args.model_name.lower():
        is_instruct = True
    else:
        is_instruct = False

    import gc
    null_num=0
    for prompt in tqdm(prompts):
    
        layer_wise_activations, head_wise_activations, _ = get_activations_bau(model, prompt, device, tokenizer, is_instruct)
        if args.token_position == "last":
            # print("last token position")
            layer_wise_activations_wanted = layer_wise_activations[:,-1,:].copy()
            head_wise_activations_wanted = head_wise_activations[:,-1,:].copy()  # last token 

        # mean
        if args.token_position =="mean":
            # print("mean token position")
            layer_wise_activations_wanted = layer_wise_activations[:,:,:].mean(axis=1).copy()
            head_wise_activations_wanted = head_wise_activations[:,:,:].mean(axis=1).copy()  #all token  

        del layer_wise_activations, head_wise_activations, _
        all_layer_wise_activations.append(layer_wise_activations_wanted)
        all_head_wise_activations.append(head_wise_activations_wanted)
 
        if np.isnan( head_wise_activations_wanted).any():
            print("null")
            null_num+=1
            continue
        gc.collect()
    print("null num:", null_num)
    # Ensure directory exists
    os.makedirs('features', exist_ok=True)
    print("Saving labels")
    np.save(f'features/{args.dataset_name}/{args.model_name}_{args.dataset_name}_{args.token_position}_labels.npy', labels)

    print("Saving layer wise activations")
    np.save(f'features/{args.dataset_name}/{args.model_name}_{args.dataset_name}_{args.token_position}_layer_wise.npy', all_layer_wise_activations)
    
    print("Saving head wise activations")
    np.save(f'features/{args.dataset_name}/{args.model_name}_{args.dataset_name}_{args.token_position}_head_wise.npy', all_head_wise_activations)
    
if __name__ == '__main__':
    main()