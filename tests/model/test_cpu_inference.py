import dataclasses
from typing import *
import torch
import torch.distributed as dist
import transformers
import torch.nn.functional as F

from realhf.base import constants, logging, testing
from realhf.impl.model.nn.real_llm_api import add_helper_functions

logger = logging.getLogger("tests.test_cpu")

def get_model_class():
    return "mistral"

def maybe_prepare_cpu_env(max_prompt_len: int):
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo", rank=0, world_size=1, init_method="tcp://localhost:7777"
        )
        import deepspeed

        deepspeed.init_distributed()
        testing.init_global_constants(
            num_dp=1,
            num_mp=1,
            num_pp=1,
            sequence_parallel=False,
            max_prompt_len=max_prompt_len,
        )
        assert dist.get_world_size() == 1, dist.get_world_size()

def get_mconfig(model_class):
    from realhf.impl.model.nn.real_llm_api import ReaLModel
    mconfig = getattr(ReaLModel, f"config_from_{model_class}")(
        model_path="/home/zzo/Quickstart/asset/model/models--mistralai--Mistral-7B-v0.3/snapshots/e676bf786d9d83284bd571f785f068e5d1f0c9f9"
    )
    return mconfig

def get_save_path():
    return "/home/zzo/Quickstart/asset/model/models--mistralai--Mistral-7B-v0.3/snapshots/e676bf786d9d83284bd571f785f068e5d1f0c9f9"

def get_cpu_real_model(model_class, mconfig, save_path):
    max_prompt_len = mconfig.n_positions
    maybe_prepare_cpu_env(max_prompt_len)
    with constants.model_scope(testing.MODEL_NAME):
        from realhf.impl.model.nn.real_llm_api import ReaLModel
        model = ReaLModel(mconfig, dtype=torch.float32, device="cpu")
        add_helper_functions(model)
        model.instantiate()
        model.eval()
    return model

def get_cpu_hf_model(save_path):
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(save_path).to(
        torch.float32
    )
    hf_model.eval()
    return hf_model

@torch.no_grad()
def test_inference_cpu_consistency(cpu_real_model, cpu_hf_model, model_class, mconfig):
    max_prompt_len = 9
    with constants.model_scope(testing.MODEL_NAME):
        bs = 1
        torch.manual_seed(1)
        input_ids = torch.tensor(
            [[1, 23325, 29576, 23045, 1066, 3415, 1136, 29576, 2]], dtype=torch.long
        )
        input_lens = torch.full((bs,), max_prompt_len, dtype=torch.int32)
        attention_mask = torch.arange(max_prompt_len)[None, :] < input_lens[:, None]

        logits1 = cpu_hf_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        logits2 = cpu_real_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

        diff = (logits1 - logits2).abs()
        print(f"mean_diff: {diff.mean()}, max_diff: {diff.max()}")

        # Compute Perplexity for cpu_hf_model
        shift_logits = logits1[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        ppl = torch.exp(loss)
        print(f"Perplexity (PPL) for cpu_hf_model: {ppl.item()}")

        # Compute Perplexity for cpu_real_model
        shift_logits_real = logits2[:, :-1, :].contiguous()
        loss_real = loss_fct(shift_logits_real.view(-1, shift_logits_real.size(-1)), shift_labels.view(-1))
        ppl_real = torch.exp(loss_real)
        print(f"Perplexity (PPL) for cpu_real_model: {ppl_real.item()}")

def main():
    # Initialize the necessary environment and configurations
    model_class = get_model_class()
    mconfig = get_mconfig(model_class)
    save_path = get_save_path()
    
    # Initialize models
    real_model = get_cpu_real_model(model_class, mconfig, save_path)
    hf_model = get_cpu_hf_model(save_path)
    
    # Run the inference consistency test
    test_inference_cpu_consistency(real_model, hf_model, model_class, mconfig)

if __name__ == "__main__":
    main()
