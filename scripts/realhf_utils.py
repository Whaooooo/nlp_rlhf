import os
import torch
import torch
import torch.distributed as dist
import transformers

try:
    from realhf.api.core.config import ModelName
    from realhf.api.core.model_api import ReaLModelConfig
    from realhf.base import constants
    from realhf.base.testing import init_global_constants
except:
    print("Can not load ReaLHF. Skipping.")
    pass

def find_latest_model(path):
    models = os.listdir(path)
    max_global_step = -1
    target = None
    for model in models:
        if 'globalstep' in model:
            version = eval(model.split('globalstep')[-1])
            if version > max_global_step:
                max_global_step = version
                target = model
    if target is None:
        raise RuntimeError(f"No suitable model found under `{path}`: {models}")
    return os.path.join(path, target)

def load_model(path, is_critic=False, device=torch.device("cuda:0"), model_family_name: str='llama'):
    # Initialize distributed environment.
    # dist.init_process_group(
    #     "nccl", rank=0, world_size=1, init_method="tcp://localhost:7777"
    # )
    model_name = ModelName("default", 0)
    init_global_constants(
        num_dp=1,
        num_mp=1,
        num_pp=1,
        sequence_parallel=False,
        model_name=model_name,
    )

    # NOTE: import here to avoid CUDA re-initialization
    from realhf.impl.model.nn.real_llm_api import ReaLModel, add_helper_functions

    # Call a method like `config_from_llama` to get the config.
    mconfig: ReaLModelConfig = getattr(ReaLModel, f"config_from_{model_family_name}")(
        transformers.AutoConfig.from_pretrained(path)
    )
    # IMPORTANT: Set the critic flag to True.
    # Since the output head and the token embedding no long have the same shape,
    # We set tied_embedding to be False.
    mconfig.is_critic = is_critic
    mconfig.tied_embedding = False

    with constants.model_scope(model_name):
        # Construct the model.
        model = ReaLModel(mconfig, dtype=torch.float16, device=device)
        model.instantiate()

        # Load the reward checkpoint
        # Since the checkpoint is already critic model, we set
        # init_critic_from_actor to be False.
        model = getattr(model, f"from_{model_family_name}")(
            path, init_critic_from_actor=False
        )
        # Add helper functions to make the model like HuggingFace models.
        add_helper_functions(model)
    return model, model_name, mconfig