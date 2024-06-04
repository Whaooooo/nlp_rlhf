SFT_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-sft-debug/20240603-1/default/epoch7epochstep5globalstep50/
RW_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-rw-debug/20240603-1/default/epoch1epochstep40globalstep40/

# python3 -m reallm.apps.quickstart sft experiment_name=quickstart-sft-debug trial_name=20240528 \
#     total_train_epochs=8 \
#     save_freq_steps=50 eval_freq_epochs=1 \
#     model.type._class=llama \
#     model.type.size=7 \
#     model.type.is_critic=False \
#     model.path=/lustre/public/pretrained_model_weights/Llama-2-7b-hf \
#     model.gradient_checkpointing=True \
#     model.optimizer.offload=False \
#     model.optimizer.type=adam \
#     dataset.train_path=/lustre/fw/datasets/imdb/rl/sft_pos-train.jsonl \
#     dataset.valid_path=/lustre/fw/datasets/imdb/rl/sft_pos-valid.jsonl \
#     dataset.max_seqlen=1024 \
#     dataset.train_tokens_per_batch=262144 \
#     dataset.valid_tokens_per_batch=262144

# python3 -m reallm.apps.quickstart rw experiment_name=quickstart-rw-debug trial_name=20240604-0 \
#     allocation_mode=pipe_model \
#     total_train_epochs=2 \
#     save_freq_steps=20 eval_freq_epochs=1 \
#     model.type._class=llama \
#     model.type.size=7 \
#     model.type.is_critic=True \
#     model.path=$SFT_MODEL_PATH \
#     model.gradient_checkpointing=True \
#     dataset.train_path=/lustre/fw/datasets/imdb/rl/rm_paired-train.jsonl \
#     dataset.valid_path=/lustre/fw/datasets/imdb/rl/rm_paired-valid.jsonl \
#     dataset.max_pairs_per_prompt=2 \
#     dataset.max_seqlen=1024 \
#     dataset.train_tokens_per_batch=131072 \
#     dataset.valid_tokens_per_batch=131072

# python3 -m reallm.apps.quickstart dpo experiment_name=quickstart-dpo-debug trial_name=20240604-0 \
#     allocation_mode=search \
#     allocation_use_cache=False \
#     n_nodes=1 \
#     n_gpus_per_node=8 \
#     recover_mode=disabled \
#     total_train_epochs=2 \
#     save_freq_steps=5 \
#     actor.type._class=llama \
#     actor.type.size=7 \
#     actor.type.is_critic=False \
#     actor.path=$SFT_MODEL_PATH \
#     actor.gradient_checkpointing=True \
#     ref.type._class=llama \
#     ref.type.size=7 \
#     ref.type.is_critic=False \
#     ref.path=$SFT_MODEL_PATH \
#     dataset.train_path=/lustre/fw/datasets/imdb/rl/rm_paired-train-lite.jsonl \
#     dataset.max_pairs_per_prompt=2 \
#     dataset.max_seqlen=512 \
#     dataset.train_tokens_per_batch=65536 \
#     dataset.valid_tokens_per_batch=65536

python3 -m reallm.apps.quickstart ppo experiment_name=remote-quickstart-ppo-debug trial_name=20240604-1 \
    allocation_mode=search \
    allocation_use_cache=False \
    n_nodes=1 \
    n_gpus_per_node=8 \
    nodelist=QH-com49 \
    recover_mode=disabled \
    save_freq_steps=null \
    global_train_bs=256 \
    global_gen_bs=256 \
    actor.type._class=llama \
    actor.type.size=7 \
    actor.type.is_critic=False \
    actor.path=$SFT_MODEL_PATH \
    actor.gradient_checkpointing=True \
    critic.type._class=llama \
    critic.type.size=7 \
    critic.type.is_critic=True \
    critic.path=$RW_MODEL_PATH \
    critic.gradient_checkpointing=True \
    ref.type._class=llama \
    ref.type.size=7 \
    ref.type.is_critic=False \
    ref.path=$SFT_MODEL_PATH \
    rew.type._class=llama \
    rew.type.size=7 \
    rew.type.is_critic=True \
    rew.path=$RW_MODEL_PATH \
    dataset.max_prompt_len=256 \
    dataset.n_tokens_per_batch=8192 \
    ppo.max_new_tokens=256 \
    ppo.min_new_tokens=256 \
    ppo.ppo_n_minibatches=4 \
    ppo.adv_norm=True ppo.value_norm=True \
    ppo.top_p=0.9 ppo.top_k=1024
