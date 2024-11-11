#!/bin/bash

python -m realhf.apps.quickstart ppo mode=local\
    experiment_name=tulu-ultrafeedback-ultrafeedback-ppo \
    trial_name=bsz64-out-scale10-bias0-temp0.7-gae1.0-kl0.05-20241028-01 \
    exp_ctrl.total_train_epochs=3 \
    exp_ctrl.save_freq_steps=308 \
    allocation_mode=pipe_model \
    actor.type._class=llama \
    actor.type.size=14 actor.backend=deepspeed \
    actor.path=/home/zzo/Quickstart/asset/model/models--allenai--tulu-2-13b/snapshots/9c8375d9f28252b499f0f14fd1e76dcbf763f9d1 \
    actor.gradient_checkpointing=True \
    actor.optimizer.warmup_steps_proportion=0.1 \
    actor.optimizer.lr=0.000001 \
    actor.optimizer.weight_decay=0.0 \
    critic.type._class=llama \
    critic.type.size=14 \
    critic.type.is_critic=True \
    critic.backend=deepspeed \
    critic.path=/home/zzo/.cache/realhf/checkpoints/zzo/tulu-ultrafeedback-rw/debug/default/epoch1epochstep117globalstep117 \
    critic.gradient_checkpointing=True \
    critic.optimizer.warmup_steps_proportion=0.1 \
    critic.optimizer.lr=0.000001 \
    critic.optimizer.weight_decay=0.0 \
    ref.type._class=llama \
    ref.type.size=14 \
    ref.backend=deepspeed \
    ref.offload=True \
    ref.path=/home/zzo/Quickstart/asset/model/models--allenai--tulu-2-13b/snapshots/9c8375d9f28252b499f0f14fd1e76dcbf763f9d1 \
    rew.type._class=llama \
    rew.type.size=14  \
    rew.type.is_critic=True \
    rew.backend=deepspeed \
    rew.offload=True \
    rew.path=/home/zzo/.cache/realhf/checkpoints/zzo/tulu-ultrafeedback-rw/debug/default/epoch1epochstep117globalstep117 \
    dataset.path=/home/zzo/Quickstart/asset/dataset/ppo/allenai___ultrafeedback_binarized_cleaned/train.json \
    dataset.max_prompt_len=1024 \
    dataset.train_bs_n_seqs=64 \
    ppo.gen.max_new_tokens=1024 \
    ppo.gen.min_new_tokens=1 \
    ppo.gen.top_p=1.0 \
    ppo.gen.temperature=0.7 \
    ppo.gen.top_k=100000000 \
    ppo.gen.force_no_logits_mask=True \
    ppo.ppo_n_minibatches=1 \
    ppo.gae_lambda=1.0 \
    actor_train.n_mbs=16 \
    actor_gen.n_mbs=1 \
    critic_train.n_mbs=16 \
    critic_inf.n_mbs=1 \
    rew_inf.n_mbs=1 \
    ref_inf.n_mbs=1 \
    ppo.kl_ctl=0.05 \
    ppo.reward_output_scaling=10.0 ppo.reward_output_bias=0.0 ppo.no_eos_penalty=-10. \
    ppo.adv_norm=True ppo.value_norm=True 