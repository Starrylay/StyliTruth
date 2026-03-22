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
import random
import time
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str,default="Qwen2.5-7B-Instruct")
parser.add_argument('--debug', type=int, default=1, help='if set, only use 100 samples for debugging')
parser.add_argument('--main_steer_style', type=str,default="None",help="主要采用的数据集的风格")
parser.add_argument('--second_steer_style', type=str,default="None",help="次要采用的数据集的风格")
parser.add_argument('--dataset_name', type=str, default="None", help='测试用的数据集')
parser.add_argument('--num_heads', type=int, default=64, help='K, number of top heads to intervene on')
parser.add_argument('--iti_intervention', type=int, default=0, help='1,0')
parser.add_argument('--main_strength', type=float, default=3.0, help='main steer strength')
parser.add_argument('--second_strength', type=float, default=3.0, help='second steer strength')
parser.add_argument('--K', type=int, default=None, help='奇异值分解保留维度')
parser.add_argument('--is_heads', type=int, default=0, help='分析实验 命名')
parser.add_argument('--token_position', type=str, default='last', help='token position to extract activations')
args = parser.parse_args()
seed = 2025    
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# # ===================== 准备参数====================
# 确定 model  

if args.model_name == "Qwen1.5-14B-Chat":
    model_path = "your own model path/Qwen1.5-14B-Chat"

model_name = args.model_name
dataset_name = args.dataset_name
dump_path = str(args.num_heads) 

tokenizer = qwen2.Qwen2Tokenizer.from_pretrained(model_path)
model = qwen2.Qwen2ForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

device = "cuda"

# # ===================== 准备测试集====================

if  dataset_name == "DRC":
    with open("dataset/Valid_DRC.json", 'r', encoding='utf-8') as file:
        data_list = json.load(file) 
elif dataset_name == "Shakespeare":
    with open("dataset/Valid_Shakespeare.json", 'r', encoding='utf-8') as file:
        data_list = json.load(file)
elif dataset_name == "tqa_gen":
    with open("dataset/Valid_tqa_gen.json", 'r', encoding='utf-8') as file:
        data_list = json.load(file)
elif dataset_name == "tqa_gen_zh":
    with open("dataset/Valid_tqa_gen_zh.json", 'r', encoding='utf-8') as file:
        data_list = json.load(file)
else:
    raise ValueError("Invalid dataset name. Please choose from ['DRC', 'Shakespeare', 'tqa_gen', 'tqa_gen_zh']")
main_steer_style = args.main_steer_style 
# ===================== 数据划分 ====================
questions = []
if args.debug == 1:
    data_list = data_list[:10]
for QA in data_list:
    questions.append(QA["question"])
answers = []
origin_answers = []

if args.main_steer_style != "None":
    # ===================== main steering vector preparation ====================
    np.load.__defaults__=(None, True, True, 'ASCII')
    probes = np.load(f"features/{main_steer_style}/{model_name}_probes_" + dump_path + f"_{args.token_position}"+ ".npy")
    top_heads = np.load(f"features/{main_steer_style}/{model_name}_top_heads_" + dump_path + f"_{args.token_position}"+".npy")
    np.load.__defaults__=(None, False, True, 'ASCII')
    with open(f"features/{main_steer_style}/{model_name}_activations_" + dump_path + f"_{args.token_position}"+".pkl", 'rb') as f:
        activations_dict = pickle.load(f)
    activations = np.load(f"features/{main_steer_style}/{model_name}_{main_steer_style}_{args.token_position}_head_wise.npy")

    num_heads = model.config.num_attention_heads
    activations = rearrange(activations, 'b l (h d) -> b l h d', h = num_heads)
    # SVD
    # import ipdb; ipdb.set_trace()
    svd_s_dict = {}
    svd_Vh_dict = {}
    for layer_no, heads in activations_dict.items():
            for head_no, vector in activations_dict[layer_no].items():
                head_activations = activations[:,layer_no,head_no,:]
                correct_activations = head_activations[::2, :]
                incorrect_activations = head_activations[1::2, :]
                correct_activations =  correct_activations - incorrect_activations
                new_s_dict, new_Vh_dict = svd_decomposition(layer_no, head_no, correct_activations)   # 全局变量
                svd_s_dict.update(new_s_dict)
                svd_Vh_dict.update(new_Vh_dict)
    print("SVD Done!")
else:
    print("No main steer vector")
K = args.K if args.K is not None else 64
# ===================== Second steer vector preparation ====================
if args.second_steer_style != "None":
    main_activations_tuple = (activations_dict, svd_s_dict, svd_Vh_dict)
    print("读取基于second_steer_style",args.second_steer_style)
    np.load.__defaults__=(None, True, True, 'ASCII')
    probes2 = np.load(f"features/{args.second_steer_style}/{model_name}_probes_" + dump_path +f"_{args.token_position}"+ ".npy")
    top_heads2 = np.load(f"features/{args.second_steer_style}/{model_name}_top_heads_" + dump_path +f"_{args.token_position}"+ ".npy")
    np.load.__defaults__=(None, False, True, 'ASCII')
    with open(f"features/{args.second_steer_style}/{model_name}_activations_" + dump_path +f"_{args.token_position}"+ ".pkl", 'rb') as f:
        activations_dict2 = pickle.load(f)     
    activations2 = np.load(f"features/{args.second_steer_style}/{model_name}_{args.second_steer_style}_{args.token_position}_head_wise.npy")
    num_heads2 = model.config.num_attention_heads
    activations2 = rearrange(activations2, 'b l (h d) -> b l h d', h = num_heads2)
    # SVD
    # import ipdb; ipdb.set_trace()
    svd_s_dict2 = {}
    svd_Vh_dict2 = {}
    for layer_no, heads in activations_dict2.items():
            for head_no, vector in activations_dict2[layer_no].items():
                head_activations2 = activations2[:,layer_no,head_no,:]
                correct_activations2 = head_activations2[::2, :]
                incorrect_activations2 = head_activations2[1::2, :]
                correct_activations2 = correct_activations2 - incorrect_activations2
                new_s_dict2, new_Vh_dict2 = svd_decomposition_2(layer_no, head_no, correct_activations2, main_activations_tuple=main_activations_tuple,K=K)   # 全局变量
                svd_s_dict2.update(new_s_dict2)
                svd_Vh_dict2.update(new_Vh_dict2)
    print("SVD Done!")
else:
    print("No second steer vector")
def get_steering_vector(layer_no, head_no, vector, cur_activations, svd_s_dict, svd_Vh_dict,activations, ):
    key = 'L' + str(layer_no) + 'H' + str(head_no)
    s = svd_s_dict[key]
    Vh = svd_Vh_dict[key]
    Vh = Vh[:K, :]
    x = vector  # delta U
    V = Vh.T 
    w = np.dot(Vh, x.T)   # belta
    w2 = np.dot(Vh, cur_activations.T) 
    head_activations = activations[:,layer_no,head_no,:]
    correct_activations = head_activations[::2, :]
    correct_activations = np.mean(correct_activations, axis=0)
    w4 = np.dot(Vh, correct_activations.T)  # w4- w2 is gamma
    w *= (1.0 + 0.5 * np.sign(w) * (w4 - w2)) 
    xx = np.dot(V, w)
    return xx


def get_activations(question):
    # import ipdb; ipdb.set_trace()
    if args.main_steer_style != "None":
        with torch.no_grad():
            for layer_no, heads in activations_dict.items():
                displacement = np.zeros((int(num_heads), int(model.config.hidden_size / num_heads)))
                device = model.model.layers[layer_no].self_attn.o_proj.weight.device.index
                displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
                # import ipdb; ipdb.set_trace()  # check model.model.layers[8].self_attn.o_proj.bias
                bias_tobe = F.linear(displacement.to(torch.float16), model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
                # import ipdb; ipdb.set_trace()  # check  model.model.layers[8].self_attn.o_proj.bias
                model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)
        # import ipdb; ipdb.set_trace()  # check  model.model.layers[8].self_attn.o_proj.bias
        prompt = question
        all_head_wise_activations = []
        layer_wise_activations, head_wise_activations, _ = get_activations_bau(model, prompt, device,tokenizer )
        all_head_wise_activations.append(head_wise_activations[:,-1,:])
        head_wise_activations = rearrange(all_head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)

        with torch.no_grad():
            for layer_no, heads in activations_dict.items():
                displacement = np.zeros((int(num_heads), int(model.config.hidden_size / num_heads)))
                for head_no, vector in activations_dict[layer_no].items():
                    cur_activations = head_wise_activations[:,layer_no,head_no,:].flatten()
                    s_vector = get_steering_vector(layer_no, head_no, vector, cur_activations, svd_s_dict, svd_Vh_dict, activations)
                    displacement[head_no] = s_vector  * args.main_strength
                device = model.model.layers[layer_no].self_attn.o_proj.weight.device.index
                displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
                bias_tobe = F.linear(displacement.to(torch.float16), model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
                model.model.layers[layer_no].self_attn.o_proj.bias += torch.nn.parameter.Parameter(bias_tobe)
    # ======================= Second =============================================
    if args.second_steer_style != "None":
        with torch.no_grad():
            for layer_no, heads  in activations_dict2.items():
                if layer_no in activations_dict:
                    continue
                displacement2 = np.zeros((num_heads2, model.config.hidden_size // num_heads2))
                device = model.model.layers[layer_no].self_attn.o_proj.weight.device.index
                disp2_t = torch.tensor(rearrange(displacement2, 'h d -> (h d)'),device=device, dtype=torch.float16)
                bias2 = F.linear(disp2_t, model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
                model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.Parameter(bias2)

        with torch.no_grad():
            for layer_no, heads in activations_dict2.items():
                displacement2 = np.zeros((num_heads2, model.config.hidden_size // num_heads2))
                for head_no, vector in activations_dict2[layer_no].items():
                    cur_activations = head_wise_activations[:,layer_no,head_no,:].flatten()
                    s_vec2 = get_steering_vector(layer_no, head_no, vector, cur_activations,svd_s_dict2, svd_Vh_dict2, activations2)
                    displacement2[head_no] = s_vec2 * args.second_strength
                device = model.model.layers[layer_no].self_attn.o_proj.weight.device.index
                disp2_t = torch.tensor(rearrange(displacement2, 'h d -> (h d)'),device=device, dtype=torch.float16)
                bias2 = F.linear(disp2_t, model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
                model.model.layers[layer_no].self_attn.o_proj.bias += torch.nn.Parameter(bias2)
    return


def my_generate(q_tokens, inputs, is_steer=True):
    generated = inputs["input_ids"]
    sequence = []
    max_length = 256
    if is_steer:
        get_activations(q_tokens)
    for i in range(max_length):
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)  # [batch, vocab]
            token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, token), dim=1)
            q_tokens = torch.cat((q_tokens, token), dim=1)
            sequence.append(token.cpu().numpy()[0][0])
            if is_steer:
                get_activations(q_tokens)
            if token.item() in {
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<|im_start|>"),
            tokenizer.convert_tokens_to_ids("<|begin_of_text|>"),
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            }:
                break
            # import ipdb; ipdb.set_trace()
            if token.cpu().numpy()[0][0] == 151643 or token.cpu().numpy()[0][0] == 151644 or token.cpu().numpy()[0][0] == 151645: 
                break
    # import ipdb; ipdb.set_trace()
    generated_text = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return generated_text


def main():

    for index, question in enumerate(questions):
        if "zh" in args.dataset_name:
            prompt = f"### Instruction:\n请你对下面的语句作出回复：\n\n### Input:\n{question}\n\n### Response:\n以下是我对该语句的回复：\n"
        else:
            prompt = f"Please respond to the following statement, and do not output any unnecessary content: \n{question}\nOkay, my answer is as follows:\n"
        inputs = tokenizer(prompt, return_tensors = 'pt')
        model_inputs = {k: v.to(model.device) for k, v in inputs.items()} 
        q_tokens = inputs.input_ids.to(model.device)
        sequence = my_generate(q_tokens, model_inputs, is_steer=True)
        print(index,question)
        print(sequence)
        answers.append(sequence)
    output_data = []
    for i in range(len(questions)):
        dict = {}
        dict["question"] = questions[i]
        dict["model_completion"] = answers[i]
        output_data.append(dict)


    # ============== 存储 ============

    fp = f"results_log/{model_name}_eval{dataset_name}_main{args.main_steer_style}_second{args.second_steer_style}_strength{args.main_strength}_{args.second_strength}_result.json"
   
    with open(fp, 'w', encoding='utf-8') as new_file:
        json.dump(output_data, new_file, ensure_ascii=False, indent=4) 


#==================================================================================================
if __name__ == "__main__":
    main()

