#!/bin/bash

python reform.py
# CUDA_VISIBLE_DEVICES=4,5,6,7 alpaca_eval evaluate --model_outputs /home/gjx/zzo_rlhf/results/sorted/mistral_dpo/mistral_dpo.json > ../logs/mistral_dpo.log
# CUDA_VISIBLE_DEVICES=4,5,6,7 alpaca_eval evaluate --model_outputs /home/gjx/zzo_rlhf/results/sorted/mistral_ppo_greedy/mistral_ppo_greedy.json > ../logs/mistral_ppo_greedy.log
# CUDA_VISIBLE_DEVICES=4,5,6,7 alpaca_eval evaluate --model_outputs /home/gjx/zzo_rlhf/results/sorted/mistral_sft_greedy/mistral_sft_greedy.json > ../logs/mistral_sft_greedy.log
CUDA_VISIBLE_DEVICES=4,5,6,7 alpaca_eval evaluate --model_outputs /home/zzo/Quickstart/alpaca_eval/my_results/mistral-tuluSftMixture-sft/debug/epoch3epochstep1globalstep4840.json > ../logs/mistral-7b-sft-beta_greedy.log