from typing import Dict, List, Optional
import torch
import transformers
from realhf.api.core.model_api import ReaLModelConfig, register_hf_family
from realhf.base.constants import use_te_impl
from realhf.base.testing import (
    TESTING_MODEL_HIDDEN_SIZE,
    TESTING_MODEL_INTERMEDIATE_SIZE,
    TESTING_MODEL_N_HEADS,
    TESTING_MODEL_N_LAYERS,
    TESTING_MODEL_N_POSITIONS,
    TESTING_MODEL_VOCAB_SIZE,
)

def convert_state_dict_armo(state_dict: Dict, config: ReaLModelConfig) -> Dict:
    new_state_dict = {}
    for k, v in state_dict.items():
        # Embedding layer
        if k == "model.embed_tokens.weight":
            new_state_dict["0.wte.weight"] = v
        # Gating network parameters
        elif k.startswith("gating."):
            new_state_dict[k] = v
        # Regression layer parameters
        elif k == "regression_layer.weight":
            new_state_dict[f"{config.n_layers + 1}.weight"] = v
        # Reward transform matrix
        elif k == "reward_transform_matrix":
            new_state_dict["reward_transform_matrix"] = v
        # Layer Norm at the end
        elif k == "model.norm.weight":
            new_state_dict[f"{config.n_layers}.ln_f.weight"] = v
        # Transformer blocks
        elif k.startswith("model.layers."):
            parts = k.split(".")
            layer_idx = int(parts[2])
            sub_module = ".".join(parts[3:])
            new_key = f"{layer_idx + 1}.{sub_module}"
            new_state_dict[new_key] = v
        else:
            # Other parameters (if any)
            new_state_dict[k] = v

    return new_state_dict

def to_armo_state_dict(state_dict: Dict[str, torch.Tensor], config: ReaLModelConfig) -> Dict:
    new_state_dict = {}
    for k, v in state_dict.items():
        # Embedding layer
        if k == "0.wte.weight":
            new_state_dict["model.embed_tokens.weight"] = v
        # Gating network parameters
        elif k.startswith("gating."):
            new_state_dict[k] = v
        # Regression layer parameters
        elif k == f"{config.n_layers + 1}.weight":
            new_state_dict["regression_layer.weight"] = v
        # Reward transform matrix
        elif k == "reward_transform_matrix":
            new_state_dict["reward_transform_matrix"] = v
        # Layer Norm at the end
        elif k == f"{config.n_layers}.ln_f.weight":
            new_state_dict["model.norm.weight"] = v
        # Transformer blocks
        elif k.startswith(tuple(str(i) for i in range(1, config.n_layers + 1))):
            parts = k.split(".")
            layer_idx = int(parts[0]) - 1
            sub_module = ".".join(parts[1:])
            new_key = f"model.layers.{layer_idx}.{sub_module}"
            new_state_dict[new_key] = v
        else:
            # Other parameters (if any)
            new_state_dict[k] = v

    return new_state_dict

def armo_embedding_layer_names(config: ReaLModelConfig) -> List[str]:
    return ["model.embed_tokens.weight"]

def armo_transformer_block_param_name(config: ReaLModelConfig, idx: int) -> List[str]:
    names = []
    layer_prefix = f"model.layers.{idx}."
    # List all possible parameter names in a transformer block
    param_names = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    names.extend([layer_prefix + name for name in param_names])
    return names

def armo_output_head_param_name(config: ReaLModelConfig) -> List[str]:
    return ["regression_layer.weight"]

def convert_config_armo(hf_config: transformers.LlamaConfig) -> ReaLModelConfig:
    # Extract custom parameters from hf_config
    config_dict = hf_config.to_dict()
    return ReaLModelConfig(
        n_layers=hf_config.num_hidden_layers,
        n_kv_heads=hf_config.num_key_value_heads,
        n_q_heads=hf_config.num_attention_heads,
        hidden_dim=hf_config.hidden_size,
        head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
        intermediate_dim=hf_config.intermediate_size,
        vocab_size=hf_config.vocab_size,
        n_positions=hf_config.max_position_embeddings,
        embd_pdrop=0.0,
        attn_pdrop=hf_config.attention_dropout,
        layer_norm_epsilon=hf_config.rms_norm_eps,
        activation_function=hf_config.hidden_act,
        use_attention_bias=hf_config.attention_bias,
        use_attn_proj_bias=hf_config.attention_bias,
        scale_attn_by_inverse_layer_idx=False,
        layer_norm_type="rms",
        mlp_type="llama",
        apply_rotary=True,
        rotary_base=hf_config.rope_theta,
        rotary_interleaved=False,
        rotary_scaling=None,
        rotary_scaling_type=None,
        # Additional custom parameters if needed
    )

def convert_config_back_armo(config: ReaLModelConfig) -> transformers.LlamaConfig:
    return transformers.LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_dim,
        intermediate_size=config.intermediate_dim,
        num_hidden_layers=config.n_layers,
        num_key_value_heads=config.n_kv_heads,
        num_attention_heads=config.n_q_heads,
        max_position_embeddings=config.n_positions,
        rms_norm_eps=config.layer_norm_epsilon,
        hidden_act=config.activation_function,
        attention_dropout=config.attn_pdrop,
        attention_bias=config.use_attention_bias,
        rope_theta=config.rotary_base,
        rope_scaling=None,
        architectures=["LlamaForRewardModelWithGating"],
        # Include custom parameters from ARMO config if necessary
        num_objectives=config.num_objectives if hasattr(config, 'num_objectives') else 19,
        gating_temperature=config.gating_temperature if hasattr(config, 'gating_temperature') else 10,
        gating_hidden_dim=config.gating_hidden_dim if hasattr(config, 'gating_hidden_dim') else 1024,
        gating_n_hidden=config.gating_n_hidden if hasattr(config, 'gating_n_hidden') else 3,
    )

def make_real_config_armo():
    hf_config = transformers.LlamaConfig(
        vocab_size=TESTING_MODEL_VOCAB_SIZE,
        max_position_embeddings=TESTING_MODEL_N_POSITIONS,
        hidden_size=TESTING_MODEL_HIDDEN_SIZE,
        intermediate_size=TESTING_MODEL_INTERMEDIATE_SIZE,
        num_hidden_layers=TESTING_MODEL_N_LAYERS,
        num_attention_heads=TESTING_MODEL_N_HEADS,
        num_key_value_heads=8,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        attention_bias=False,
        rope_theta=500000.0,
        architectures=["LlamaForRewardModelWithGating"],
        num_objectives=19,
        gating_temperature=10,
        gating_hidden_dim=1024,
        gating_n_hidden=3,
    )
    return convert_config_armo(hf_config)

# Register the ARMO model
register_hf_family(
    name="armo",
    hf_cls_name="LlamaForRewardModelWithGating",
    config_from_hf_converter=convert_config_armo,
    config_to_hf_converter=convert_config_back_armo,
    sd_from_hf_converter=convert_state_dict_armo,
    sd_to_hf_converter=to_armo_state_dict,
    embedding_param_names=armo_embedding_layer_names,
    tblock_param_names=armo_transformer_block_param_name,
    head_param_names=armo_output_head_param_name,
    real_config_maker=make_real_config_armo,
)
