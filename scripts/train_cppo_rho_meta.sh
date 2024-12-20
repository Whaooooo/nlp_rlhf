python -m realhf.apps.quickstart cppo \
    mode=local \
    n_nodes=1 \
    n_gpus_per_node=2 \
    experiment_name=rho-no-prompt-meta-kimimeta-meta-cppo \
    trial_name=temp1.0-lambda1.0-scale100-bias0.5-kl0.005-test2 \
    exp_ctrl.total_train_epochs=1 \
    exp_ctrl.save_freq_steps=116 \
    allocation_mode=heuristic \
    actor.type._class=llama \
    actor.type.size=1 \
    actor.backend=megatron \
    actor.path=/root/autodl-tmp/realhf/checkpoints/root/rho-no-prompt-meta-kimimeta-sft/debug/default/epoch20epochstep1globalstep133 \
    actor.gradient_checkpointing=True \
    actor.optimizer.warmup_steps_proportion=0.1 \
    actor.optimizer.weight_decay=0.0 \
    actor.optimizer.lr=0.000001 \
    ref.backend=megatron \
    ref.path=/root/autodl-tmp/realhf/checkpoints/root/rho-no-prompt-meta-kimimeta-sft/debug/default/epoch20epochstep1globalstep133 \
    critic.type._class=llama \
    critic.type.size=1 \
    critic.type.is_critic=True \
    critic.backend=megatron \
    critic.path=/root/autodl-tmp/realhf/checkpoints/root/rho-no-prompt-meta-kimimeta-sft/debug/default/epoch20epochstep1globalstep133 \
    critic.init_critic_from_actor=True \
    critic.gradient_checkpointing=True \
    critic.optimizer.warmup_steps_proportion=0.1 \
    critic.optimizer.weight_decay=0.0 \
    critic.optimizer.lr=0.000001 \
    dataset.path=/root/autodl-tmp/dataset/sft/math_or_code/meta-math___meta_math_qa/meta-math___meta_math_qa_test.json \
    dataset.max_prompt_len=512 \
    dataset.train_bs_n_seqs=256 \
    cppo.gen.max_new_tokens=1536 \
    cppo.gen.min_new_tokens=1 \
    cppo.gen.top_p=1.0 \
    cppo.gen.temperature=1.0 \
    cppo.gen.top_k=20000000 \
    cppo.gen.force_no_logits_mask=True \
    cppo.cppo_n_minibatches=1 \
    actor_train.n_mbs=16 \
    actor_gen.n_mbs=1 \
    critic_train.n_mbs=16 \
    critic_inf.n_mbs=1 \
    cppo.kl_ctl=0.005 \
    cppo.reward_output_scaling=100 \
    cppo.reward_output_bias=0.5 \
    cppo.gae_lambda=1.0 \
    cppo.adv_norm=True cppo.value_norm=True \
    actor_train.parallel.pipeline_parallel_size=1 \
    actor_train.parallel.model_parallel_size=1 \
    actor_train.parallel.data_parallel_size=1 \
    critic_train.parallel.pipeline_parallel_size=1 \
    critic_train.parallel.model_parallel_size=1 \
    critic_train.parallel.data_parallel_size=1 \
    actor_gen.parallel.pipeline_parallel_size=1 \
    actor_gen.parallel.model_parallel_size=1 \
    actor_gen.parallel.data_parallel_size=1 \
    critic_inf.parallel.pipeline_parallel_size=1 \
    critic_inf.parallel.model_parallel_size=1 \
    critic_inf.parallel.data_parallel_size=1 \
    ref_inf.parallel.pipeline_parallel_size=1 \
    ref_inf.parallel.model_parallel_size=1 \
    ref_inf.parallel.data_parallel_size=1 \