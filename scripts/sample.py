import os
import gc
import json
import tqdm
import time

# run `export PYTHONPATH=$PYTHONPATH:PATH_TO_REALHF_DIRECTORY` first
import realhf_utils
from realhf.base import constants

import torch
import numpy as np

from argparse import ArgumentParser
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# PROMPT_USER: str = 'BEGINNING OF CONVERSATION: USER: {input}'
PROMPT_USER: str = '<|user|>\n{input}</s>'
# PROMPT_ASSISTANT: str = 'ASSISTANT' 
PROMPT_ASSISTANT: str = '\n<|assistant|>\n'

if __name__ == "__main__":
    # policy_path = "/home/zzo/Quickstart/asset/model/models--allenai--tulu-2-13b/snapshots/9c8375d9f28252b499f0f14fd1e76dcbf763f9d1" # tulu 2 13b base model
    # policy_path = "/home/zzo/.cache/realhf/checkpoints/zzo/tulu-ultrafeedback-ultrafeedback-ppo/debug/actor/epoch2epochstep1globalstep924" # tulu2_13b_ppo_1024
    # policy_path = "/home/zzo/.cache/realhf/checkpoints/zzo/tulu-ultrafeedback-ultrafeedback-ppo/bsz64-out-scale10-bias0-temp0.7-gae1.0-kl0.035/actor/epoch2epochstep427globalstep1350" # tulu2_13b_ppo_1026
    # policy_path = "/home/zzo/.cache/realhf/checkpoints/zzo/mistral-ultrachat-ultrafeedback-ultrafeedback-ppo/test10-lambda095/actor/epoch2epochstep358globalstep1200" # mistral-ppo
    # policy_path = "/home/zzo/.cache/realhf/checkpoints/zzo/mistral-ultrachat-sft/debug/default/epoch1epochstep1280globalstep1280" #  mistral-sft
    policy_path = "/home/zzo/.cache/realhf/checkpoints/root/tulu-ultrafeedback-dpo/debug/actor/epoch4epochstep1globalstep5422" # mistral-7b-sft-beta
    # policy_path = "/home/zzo/.cache/realhf/checkpoints/zzo/tulu-ultrafeedback-dpo/debug/actor/epoch4epochstep1globalstep5698/" # tulu2_13b_dpo
    # reward_path = "/home/zzo/.cache/realhf/checkpoints/zzo/tulu-ultrafeedback-rw/debug/default/epoch1epochstep117globalstep117" # tulu2_13b
    reward_path = "/home/zzo/.cache/realhf/checkpoints/root/mistral-tuluSftMixture-tulu2.5-rw/debug/default/epoch1epochstep111globalstep111" # mistral
    data_path='/home/zzo/Quickstart/alpaca_eval/results/gpt4_1106_preview/model_outputs.json'
    # data_path='/home/zzo/Quickstart/asset/dataset/rw/allenai___ultrafeedback_binarized_cleaned/train.json' #  train
    save_path='/home/zzo/Quickstart/alpaca_eval/my_results/mistral_tulu-2.5-dpo/debug/actor/epoch4epochstep1globalstep5422.json'
    device = "cuda" # the device to load the model onto

    # load test data
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    if data_path.endswith('train.json'):
        np.random.seed(10)
        np.random.shuffle(data)
        data = data[:1000]

    print("{} samples".format(len(data)))

    # load policy model
    policy_model = LLM(model=policy_path, 
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.95,
                    swap_space=0.0,
                    max_num_seqs=32)
    sampling_params = SamplingParams(temperature=1.0, 
                                     max_tokens=1024,
                                     # top_p=1.0,
                                     top_k=1,
                                     n=1)
    # load reward model
    reward_model, rw_name, rw_mconfig = realhf_utils.load_model(reward_path, is_critic=True, model_family_name='mistral', device=torch.device('cuda:1'))
    reward_model.eval()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(reward_path)
    # tokenizer.padding_side = "right"

    # prepare prompts
    prompts = []
    for d in data:
        if 'prompt' in d:
            prompt = d['prompt']
        else:
            instruction = d['instruction']
            
            prompt = PROMPT_USER.format(input=instruction) + PROMPT_ASSISTANT
        
            # prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': instruction}], add_generation_prompt=True, tokenize=False)

        prompts.append(prompt)

    print(prompts[0])
    
    outputs = policy_model.generate(prompts, sampling_params)

    print(outputs[0].outputs[0].text)

    # reward prompts
    reward_prompts = []
    for output in outputs:
        for out_idx in range(len(output.outputs)):
            reward_prompts.append(output.prompt + output.outputs[out_idx].text + tokenizer.eos_token)

    print("{} reward prompts".format(len(reward_prompts)), flush=True)

    # compute rewards    
    with constants.model_scope(rw_name):
        inputs = tokenizer(reward_prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)

        device = torch.device('cuda:1')
        input_ids = inputs['input_ids'].to(device)
        attention_mask =  inputs['attention_mask'].to(device)
        rw_scores = []
        eval_size = 16
        for i in tqdm.tqdm(range(0, len(reward_prompts), eval_size)):
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            rw_scores.append(reward_model(input_ids[i:i+eval_size], attention_mask[i:i+eval_size]).logits.detach().cpu())
        rw_scores = torch.cat(rw_scores, dim=0)
        # for r, attn_mask in zip(rw_scores, attention_mask):
        #     print(r[-attn_mask.sum():, 0], flush=True)
        rw_scores = [r[-1, 0].item() for r, attn_mask in zip(rw_scores, attention_mask)]
        assert len(reward_prompts) == len(rw_scores)
    
    all_rewards = []
    all_token_lengths = []

    for d, output in zip(data, outputs):
        outputs_w_rewards = []
        for out_idx in range(len(output.outputs)):
            outputs_w_rewards.append((output.outputs[out_idx].text, rw_scores.pop(0)))
        rewards = [r for o, r in outputs_w_rewards]
        print("Rewards:", rewards)

        w = np.argmax(rewards)
        o = outputs_w_rewards[w][0]

        d['output'] = o
        d['generator'] = policy_path

        all_rewards.append(np.max(rewards))
        all_token_lengths.append(len(tokenizer(output.prompt + d['output'])['input_ids']))


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    json.dump(data, open(save_path, 'w', encoding='utf-8'), indent=4)

    print(policy_path, flush=True)
    print("Saved to {}".format(save_path))
    print("Avg. Rewards:", np.mean(all_rewards))
    print("Avg. Seq Length:", np.mean(all_token_lengths))