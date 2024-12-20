python -m realhf.apps.quickstart clip-sft \
    experiment_name=rho-meta-clip-sft \
    trial_name=lr0.00001-clipratio0.997-test3 \
    allocation_mode=manual \
    mode=local \
    n_nodes=1 \
    n_gpus_per_node=1 \
    exp_ctrl.total_train_epochs=1 \
    exp_ctrl.save_freq_steps=306 \
    exp_ctrl.eval_freq_steps=306 \
    model.type._class=llama \
    model.backend=megatron \
    model.path=/root/autodl-tmp/base_model/models--microsoft--rho-math-1b-v0.1/snapshots/0b1db8d22b330c281cc810899d7938f023d78195 \
    model.optimizer.lr=0.00001 \
    model.optimizer.weight_decay=0.0 \
    model.optimizer.warmup_steps_proportion=0.1 \
    clip_ratio=0.997 \
    dataset.train_path=/root/autodl-tmp/dataset/sft/math_or_code/meta-math___meta_math_qa/9-1_train.jsonl \
    dataset.valid_path=/root/autodl-tmp/dataset/sft/math_or_code/meta-math___meta_math_qa/9-1_test.jsonl \
    dataset.max_seqlen=2048 \
    allocation.n_mbs=16 \
    allocation.parallel.pipeline_parallel_size=1 \
    allocation.parallel.model_parallel_size=1 \
    allocation.parallel.data_parallel_size=1 \
    dataset.train_bs_n_seqs=1024 \
    dataset.valid_bs_n_seqs=128