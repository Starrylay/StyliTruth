# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import torch
# import numpy as np
# import pickle
# import sys

# sys.path.append('../')
# from utils import get_activations_bau, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q
# from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions, svd_decomposition,svd_decomposition_2
# import llama
# import qwen2
import argparse
import json
# from tqdm import tqdm
# from einops import rearrange
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import math
# from transformers import AutoModelForCausalLM, AutoTokenizer


from transformers import AutoTokenizer
import torch
import qwen2
import llama



parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str,default="Qwen2.5-7B")
parser.add_argument('--debug', type=int, default=1, help='if set, only use 100 samples for debugging')
parser.add_argument('--main_steer_style', type=str,default="DRC",help="主要采用的数据集的风格")
parser.add_argument('--second_steer_style', type=str,default="None",help="次要采用的数据集的风格")
parser.add_argument('--dataset_name', type=str, default="DRC", help='测试用的数据集')
parser.add_argument('--num_heads', type=int, default=64, help='K, number of top heads to intervene on')
args = parser.parse_args()



# # ===================== 准备参数====================
# 确定 model  
if args.model_name == "Qwen2.5-7B-Instruct":
    model_path = "/new_disk1/chenglei_shen/projects/PretrainModels/Qwen2.5-7B-Instruct"
elif args.model_name == "Llama-3-8B-Instruct":
    model_path = "/new_disk1/chenglei_shen/projects/PretrainModels/Llama-3-8B-Instruct" 
elif args.model_name == "Qwen1.5-14B-Chat":
    model_path = "/new_disk1/chenglei_shen/projects/PretrainModels/Qwen1.5-14B-Chat"
elif args.model_name == "Qwen2.5-7B":
    model_path = "/new_disk1/chenglei_shen/projects/PretrainModels/Qwen2.5-7B"
elif args.model_name == "Llama-3-8B-Instruct":
    model_path = "/new_disk1/chenglei_shen/projects/PretrainModels/Llama-3-8B-Instruct"
elif args.model_name == "Llama-3.2-3B-Instruct":
    model_path  = "/new_disk1/chenglei_shen/projects/PretrainModels/Llama-3.2-3B-Instruct"
elif args.model_name == "Qwen2.5-3B-Instruct":
    model_path = "/new_disk1/chenglei_shen/projects/PretrainModels/Qwen2.5-3B-Instruct"
elif args.model_name == "Qwen2-7B":
    model_path = "/new_disk1/chenglei_shen/projects/PretrainModels/Qwen2-7B"
elif args.model_name == "Qwen2-7B-Instruct":
    model_path = "/new_disk1/chenglei_shen/projects/PretrainModels/Qwen2-7B-Instruct"
elif args.model_name == "Llama-3-8B":
    model_path = "/new_disk1/chenglei_shen/projects/PretrainModels/Llama-3-8B"
model_name = args.model_name


dataset_name = args.dataset_name

print(args.num_heads)
dump_path = str(args.num_heads)  # 64_3.0
print(dump_path)    # 64_3.0

# # ===================== 准备模型====================
if "qwen" in args.model_name.lower():

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = qwen2.Qwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.float32, device_map="auto")

    
    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.bfloat32,
    #     device_map="auto",
    # )

elif "llama" in args.model_name.lower():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = llama.LlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.float32, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

device = "cuda"
model.eval()
# tokenizer = AutoTokenizer.from_pretrained(args.model_path)
# model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True, torch_dtype=torch.float32, device_map="auto")

# # ===================== 准备测试集====================

if  dataset_name == "DRC":
    # main_steer_style = "DRC"   # 无视控制风格，采用测试集数据风格
    with open("dataset/Valid_DRC.json", 'r', encoding='utf-8') as file:
        data_list = json.load(file)
elif  dataset_name == "DRC_merge":
    # main_steer_style = "DRC"   # 无视控制风格，采用测试集数据风格
    with open("dataset/Train_DRC_merge.json", 'r', encoding='utf-8') as file:
        data_list = json.load(file)
elif  dataset_name == "DRC_白话_question":
    # main_steer_style = "DRC"   # 无视控制风格，采用测试集数据风格
    with open("dataset/Train_DRC_白话_question.json", 'r', encoding='utf-8') as file:
        data_list = json.load(file)  
elif  dataset_name == "DRC_tqa_question":
    # main_steer_style = "DRC"   # 无视控制风格，采用测试集数据风格
    with open("dataset/Train_DRC_tqa_question.json", 'r', encoding='utf-8') as file:
        data_list = json.load(file)  
elif dataset_name == "Shakespeare":
    # main_steer_style = "Shakespeare" # 无视控制风格，采用测试集数据风格
    with open("dataset/Valid_Shakespeare.json", 'r', encoding='utf-8') as file:
        data_list = json.load(file)
elif dataset_name == "tqa_gen":
    # main_steer_style = args.main_steer_style # 采用超参数风格
    with open("dataset/Valid_tqa_gen.json", 'r', encoding='utf-8') as file:
        data_list = json.load(file)
elif dataset_name == "tqa_gen_zh":
    with open("dataset/Valid_tqa_gen_zh.json", 'r', encoding='utf-8') as file:
        data_list = json.load(file)
elif dataset_name == "tqa_gen_zh_文言":
    with open("dataset/Valid_tqa_gen_zh_文言.json", 'r', encoding='utf-8') as file:
        data_list = json.load(file)

main_steer_style = args.main_steer_style # 采用超参数风格
# ===================== 数据划分 ====================
questions = []
if args.debug == 1:
    data_list = data_list[:10]
for QA in data_list:
    questions.append(QA["question"])
answers = []
# 读取基于steer_style计算并保存好的steer向量

# 读取干预后的结果
# import  ipdb; ipdb.set_trace()

# def my_generate(q_tokens, inputs):
#     generated = inputs["input_ids"]
#     sequence = []
#     max_length = 600

#     for i in range(max_length):
#         with torch.no_grad():
#             outputs = model(generated)
#             next_token_logits = outputs.logits[:, -1, :]
#             probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
#             token = torch.tensor([probs.argmax().item()]).unsqueeze(0).to('cuda:0')
#             generated = torch.cat((generated, token), dim=1)
#             q_tokens = torch.cat((q_tokens, token), dim=1)
#             sequence.append(token.cpu().numpy()[0][0])
#             # import ipdb; ipdb.set_trace()
#             if token.item() in {
#             tokenizer.convert_tokens_to_ids("<|endoftext|>"),
#             tokenizer.convert_tokens_to_ids("<|im_end|>"),
#             tokenizer.convert_tokens_to_ids("<|im_start|>")
#             }:
#                 break
#     generated_text = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
#     return generated_text


# from transformers.generation.logits_process import (
#     LogitsProcessorList,
#     RepetitionPenaltyLogitsProcessor,
#     NoRepeatNGramLogitsProcessor,
#     TemperatureLogitsWarper,
#     TopPLogitsWarper,
# )
# from transformers.generation.stopping_criteria import StoppingCriteriaList



# @torch.inference_mode()
# def my_generate_like_hf(
#     model,
#     tokenizer,
#     inputs,                             # {"input_ids": ..., "attention_mask": ...}
#     max_new_tokens=256,
#     do_sample=False,
#     temperature=1.0,
#     top_p=1.0,
#     repetition_penalty=1.0,
#     no_repeat_ngram_size=0,
#     eos_token_id=None,                  # 可传 int 或 [int, ...]
#     pad_token_id=None,
#     logits_processor: LogitsProcessorList = None,
#     logits_warper: LogitsProcessorList = None,
#     stopping_criteria: StoppingCriteriaList = None,
#     on_token=None,                      # 可选回调：on_token(token_id, step)
# ):
#     model.eval()

#     # ---- 准备输入/设备 ----
#     input_ids = inputs["input_ids"]
#     attn_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
#     device = input_ids.device
#     generated = input_ids
#     past_kv = None
#     new_token_ids = []

#     # ---- 停止符集合 ----
#     stop_ids = set()
#     if eos_token_id is not None:
#         # import ipdb; ipdb.set_trace()
#         if isinstance(eos_token_id, (list, tuple, set)):
#             stop_ids.update(int(x) for x in eos_token_id)
#         else:
#             stop_ids.add(int(eos_token_id))
#     if tokenizer.eos_token_id is not None:
#         stop_ids.add(int(tokenizer.eos_token_id))
#     # 尝试加入 <|eot_id|> / <|im_end|> 等
#     for tok in ("<|eot_id|>", "<|im_end|>", "<|endoftext|>"):
#         try:
#             tid = tokenizer.convert_tokens_to_ids(tok)
#             if tid is not None and tid != tokenizer.unk_token_id:
#                 stop_ids.add(int(tid))
#         except Exception:
#             pass
#     if pad_token_id is None:
#         pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

#     # ---- 组装 processors/warpers ----
#     procs = LogitsProcessorList()
#     if logits_processor is not None:
#         procs.extend(logits_processor)
#     if repetition_penalty and repetition_penalty != 1.0:
#         procs.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
#     if no_repeat_ngram_size and no_repeat_ngram_size > 0:
#         procs.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))

#     warpers = LogitsProcessorList()
#     if logits_warper is not None:
#         warpers.extend(logits_warper)
#     # 温度/Top-p 的实现放在 warper 里，和 HF 一致
#     if do_sample:
#         if temperature and temperature != 1.0:
#             warpers.append(TemperatureLogitsWarper(temperature=temperature))
#         if top_p and top_p < 1.0:
#             warpers.append(TopPLogitsWarper(top_p=top_p))

#     stops = stopping_criteria or StoppingCriteriaList([])

#     # ---- 主循环（带 KV cache）----
#     for step in range(max_new_tokens):
#         step_input = generated if past_kv is None else next_token  # 首步喂全序列，后续只喂上一个 token
#         outputs = model(
#             input_ids=step_input,
#             attention_mask=attn_mask if past_kv is None else None,
#             use_cache=True,
#             past_key_values=past_kv,
#         )
#         past_kv = outputs.past_key_values

#         logits = outputs.logits[:, -1, :]  # (bs=1, vocab)
#         # 防御性处理，避免极端数值造成 NaN/Inf
#         logits = torch.nan_to_num(logits, neginf=-1e4, posinf=1e4)

#         # 先过 processors（重复惩罚/不重复 ngram 等）
#         if len(procs) > 0:
#             logits = procs(generated, logits)

#         # 选择策略：采样 or 贪心
#         if do_sample:
#             # 过 warpers（温度/Top-p）
#             if len(warpers) > 0:
#                 logits = warpers(generated, logits)
#             probs = torch.softmax(logits, dim=-1)
#             probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
#             next_token = torch.multinomial(probs, num_samples=1)  # (bs=1,1)
#         else:
#             next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (bs=1,1)

#         # 追加到序列与 mask
#         generated = torch.cat([generated, next_token], dim=1)
#         attn_mask = torch.cat([attn_mask, torch.ones_like(next_token)], dim=1)
#         new_id = int(next_token.item())
#         new_token_ids.append(new_id)

#         # 可选回调
#         if on_token is not None:
#             try:
#                 on_token(new_id, step)
#             except Exception:
#                 pass

#         # 停止条件：命中 stop id 或自定义 stopping_criteria 返回 True
#         if new_id in stop_ids:
#             break
#         if stops(generated, logits):
#             break

#     # ---- 解码新增部分 ----
#     prompt_len = input_ids.size(1)
#     text = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)

#     # 右侧 pad（可选）
#     if generated.size(1) < prompt_len + max_new_tokens:
#         pad_len = prompt_len + max_new_tokens - generated.size(1)
#         pad_ids = torch.full((1, pad_len), pad_token_id, dtype=generated.dtype, device=device)
#         generated = torch.cat([generated, pad_ids], dim=1)

#     return text, generated, new_token_ids


def my_generate(q_tokens, inputs):
    generated = inputs["input_ids"]
    sequence = []
    max_length = 600

    for i in range(max_length):
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            token = torch.tensor([probs.argmax().item()]).unsqueeze(0).to('cuda:0')
            generated = torch.cat((generated, token), dim=1)
            q_tokens = torch.cat((q_tokens, token), dim=1)
            sequence.append(token.cpu().numpy()[0][0])
            # import ipdb; ipdb.set_trace()
            
            if token.item() in {
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<|im_start|>"),

            tokenizer.convert_tokens_to_ids("<|begin_of_text|>"),
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
         
            }:
                # print(token.item())
                break
    generated_text = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return generated_text



def main():
    # import ipdb; ipdb.set_trace()
    if "llama" in args.model_name.lower():
        if "DRC" in args.dataset_name :
            systeam_content = "请你以对话的形式直接对下面的语句作出回应，用中文回答，不要输出不必要的内容。"
        elif args.dataset_name == "tqa_gen":
            systeam_content = "Please respond to the following statement, and do not output any unnecessary content:"
        elif "tqa_gen_zh" in args.dataset_name:
            systeam_content = "请你以对话的形式直接对下面的语句作出回应，用中文回答，不要输出不必要的内容。"
        elif args.dataset_name == "Shakespeare":
            systeam_content = "Please respond to the following statement, and do not output any unnecessary content:"
    else:
        if "DRC" in args.dataset_name :
            systeam_content = "请你以对话的形式直接对下面的语句作出回应，不要输出不必要的内容。"
        elif args.dataset_name == "tqa_gen":
            systeam_content = "Please respond to the following statement, and do not output any unnecessary content:"
        elif "tqa_gen_zh" in args.dataset_name:
            systeam_content = "请你以对话的形式直接对下面的语句作出回应，不要输出不必要的内容。"
        elif args.dataset_name == "Shakespeare":
            systeam_content = "Please respond to the following statement, and do not output any unnecessary content:"
    
    for index, question in enumerate(questions):
        if "instruct" in args.model_name.lower():
            messages =  [
                            {"role": "system", "content": systeam_content},
                            {"role": "user", "content": f"{question}"},
                        ] 
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer(prompt_str, return_tensors='pt').to(model.device)
            q_tokens = model_inputs.input_ids
        else:
                # import ipdb; ipdb.set_trace()
            if "zh" in args.dataset_name:
                prompt = f"请对以下问题做出回复不要输出多余的内容：\n问题：{question}\n回复："
            else:
                prompt = f"Please respond to the following statement, and do not output any unnecessary content:\nQuestion: {question}\nAnswer: "
            # question = "如果你走进一个衣柜会发生什么？"
            prompt = f"### Instruction:\n请你对下面的语句作出回复：\n\n### Input:\n{question}\n\n### Response:\n以下是我对该语句的回复：\n"
            # prompt = f"请对以下问题做出回复不要输出多余的内容：\n问题：{question}\n 回复："
        
            inputs = tokenizer(prompt, return_tensors="pt")
            model_inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # sequence, full_ids, new_ids = my_generate_like_hf(
            #     model, tokenizer, model_inputs,
            #     max_new_tokens=512,
            #     do_sample=False,               # =贪心；若采样改 True
            #     temperature=0.7, top_p=0.9,   # 仅在 do_sample=True 时生效
            #     repetition_penalty=1.05,
            #     no_repeat_ngram_size=6,
            #     eos_token_id=[tokenizer.eos_token_id],
            #     pad_token_id=tokenizer.eos_token_id,
            # )
        
            q_tokens = inputs.input_ids.to(model.device)
        sequence = my_generate(q_tokens, model_inputs)
        # sequence = model.generate(**model_inputs,max_new_tokens=256,do_sample=False)
        # sequence = tokenizer.decode(sequence[0][model_inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
      

        print("=======================================")
        print(index,question)
        print(sequence)

        answers.append(sequence)
        # answer = tokenizer.decode(sequence, skip_special_tokens=True)
        # print(answer)
    output_data = []
    for i in range(len(questions)):
        dict = {}
        dict["question"] = questions[i]
        dict["origin_model_completion"] = answers[i]   
        # dict["model_path"] = model_path
        output_data.append(dict)
    # # ============== 存储 ============
    with open(f"results_log/{model_name}_eval{dataset_name}_origin_result.json", 'w', encoding='utf-8') as new_file:
        json.dump(output_data, new_file, ensure_ascii=False, indent=4)

#==================================================================================================
if __name__ == "__main__":
    main()

