from typing import Callable, List, Optional, Union
import dataclasses
import itertools
import json

from transformers.activations import GELUActivation
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from impl.model.backend.ds_pipe_engine import LayerSpec, PipelineModule
from impl.model.backend.ds_pipe_engine.topology import PipeDataParallelTopology
from impl.model.utils.data import (mask_eos_token, TensorDataclassToTupleInterface, upcast_masked_softmax,
                                   upcast_softmax)
from impl.model.utils.logits_warper import top_k_top_p_logits
import api.huggingface
import api.model


@dataclasses.dataclass
class TransformerConfig:
    n_layers: int
    n_heads: int
    head_dim: int
    hidden_dim: int
    intermediate_dim: int  # for mlp, usually 4*h
    vocab_size: int
    n_positions: int
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu"

    @staticmethod
    def from_huggingface_config(config_file):
        config_json = json.load(open(config_file, "r"))
        # for starcoder config
        config = TransformerConfig(n_layers=config_json["n_layer"],
                                   n_heads=config_json["n_head"],
                                   head_dim=config_json["n_embd"] // config_json["n_head"],
                                   hidden_dim=config_json["n_embd"],
                                   intermediate_dim=config_json["n_inner"],
                                   vocab_size=config_json["vocab_size"],
                                   n_positions=config_json["n_positions"],
                                   resid_pdrop=config_json["resid_pdrop"],
                                   attn_pdrop=config_json["attn_pdrop"],
                                   embd_pdrop=config_json["embd_pdrop"],
                                   layer_norm_epsilon=config_json["layer_norm_epsilon"],
                                   activation_function="gelu")
        return config


@dataclasses.dataclass
class TransformerData(TensorDataclassToTupleInterface):
    # input, and config
    raw_input_ids: torch.Tensor = None
    raw_attention_mask: torch.Tensor = None
    input_ids: torch.Tensor = None
    attention_mask: torch.Tensor = None
    labels: torch.Tensor = None
    position_ids: torch.Tensor = None
    hidden_states: torch.Tensor = None
    kv_cache: torch.Tensor = None
    head_mask: torch.Tensor = None
    generation_id: int = None  # for kv cache, should > 0
    # outputs
    loss: torch.Tensor = None
    logits: torch.Tensor = None


class MQATransformerForCausalLM(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.preprocess = Preprocess(config.n_positions)
        self.transformer = MQATransformer(config)
        # self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.postprocess = Postprocess(config.hidden_dim, config.vocab_size)

        self.layers = self.to_layers()

    def load_from_path(self, model_path, device, dtype):
        try:
            state_dict = torch.load(model_path)
        except IsADirectoryError:
            import os
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))

        new_state_dict = {}
        for k, v in state_dict.items():
            # TODO: converter for each model, following is for starcoder
            replace_from = [".wte", ".wpe", ".ln_f", "lm_head"]
            replace_to = [
                ".embedding_layer.wte", ".embedding_layer.wpe", ".ln_f.inner", "postprocess.lm_head"
            ]
            for rf, rt in zip(replace_from, replace_to):
                if rf in k:
                    k = k.replace(rf, rt)
            new_state_dict[k] = v

        self.load_state_dict(new_state_dict)
        self.to(dtype=dtype, device=device)

    def generate(self, *args, **kwargs):
        # TODO: support huggingface generate
        return generate(self, *args, **kwargs)

    def to_layers(self):
        return [self.preprocess] + self.transformer.to_layers() + [self.postprocess]

    def forward(self, x: TransformerData):
        # inputs should be a tuple (input_ids, attention_mask, position_ids, generation_id)
        # TODO: check if generation_id could be an int or None
        for l in self.layers:
            x = l(x)
        return x


class MQATransformer(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embedding_layer = VocabPositionEmbedding(config.vocab_size, config.n_positions,
                                                      config.hidden_dim)
        self.h = nn.ModuleList([MQATransformerBlock(config, i) for i in range(config.n_layers)])
        self.ln_f = LastLayerNorm(config.hidden_dim, config.layer_norm_epsilon)

    def to_layers(self):
        return [self.embedding_layer] + list(self.h) + [self.ln_f]

    def forward(self, x: TransformerData):
        x = self.embedding_layer(x)
        for layer in self.h:
            x = layer(x)
        x = self.ln_f(x)
        return x


class Preprocess(nn.Module):

    def __init__(self, n_positions):
        super().__init__()
        self.generation_id = 0  # for generate
        self.generation_length_cache = dict()
        self.register_buffer("bias",
                             torch.tril(torch.ones((n_positions, n_positions), dtype=torch.bool)),
                             persistent=False)

    # wrap pipe: input tensors
    def forward(self, x: TransformerData):
        if x.input_ids is None:
            x.input_ids = x.raw_input_ids
        assert len(x.input_ids.shape) == 2, "input_ids should be batched input sequence as a 2-d tensor"
        # if x.attention_mask is not None:
        #     assert x.input_ids.shape == x.attention_mask.shape, "input_ids and attention_mask should have the same shape"
        batch_size, input_length = x.input_ids.shape
        device = x.input_ids.device

        if x.generation_id is None:
            x.generation_id = self.generation_id
            self.generation_length_cache[self.generation_id] = input_length
            self.generation_id += 1
            past_length = 0
        else:
            past_length = self.generation_length_cache[x.generation_id]
            self.generation_length_cache[x.generation_id] += input_length

        if x.position_ids is None and x.attention_mask is not None:
            x.position_ids = x.attention_mask.long().cumsum(-1) - 1
            x.position_ids.masked_fill_(x.attention_mask == 0, 1)
            if past_length > 0:
                x.position_ids = x.position_ids[:, past_length:input_length + past_length]
            x.position_ids.to(device=device, dtype=torch.long)
        elif x.position_ids is None:
            x.position_ids = torch.arange(past_length,
                                          input_length + past_length,
                                          dtype=torch.long,
                                          device=device)
            x.position_ids = x.position_ids.unsqueeze(0).view(-1, input_length)

        # attention mask: lower triangle of [1, s, 1, s]
        mask = self.bias[past_length:past_length + input_length, :past_length + input_length]

        if x.raw_attention_mask is None:
            x.attention_mask = mask.unsqueeze(0).unsqueeze(2).to(dtype=torch.bool, device=device)
        else:
            x.attention_mask = mask * x.raw_attention_mask.unsqueeze(1)
            x.attention_mask = x.attention_mask.unsqueeze(2).to(dtype=torch.bool, device=device)

        return x


class Postprocess(nn.Module):

    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x: TransformerData):
        lm_logits = self.lm_head(x.hidden_states)
        loss = None
        if x.labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = x.labels[..., 1:].contiguous().to(shift_logits.device)
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        x.logits = lm_logits
        x.loss = loss
        return x


class LastLayerNorm(nn.Module):

    def __init__(self, hidden_dim, eps):
        super().__init__()
        self.inner = nn.LayerNorm(hidden_dim, eps=eps)

    def forward(self, x: TransformerData):
        x.hidden_states = self.inner(x.hidden_states)
        return x


class VocabPositionEmbedding(nn.Module):

    def __init__(self, vocab_size, n_positions, hidden_dim):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden_dim)
        self.wpe = nn.Embedding(n_positions, hidden_dim)

    def forward(self, x: TransformerData):
        # x includes all inputs of the model, forward function of nn will be passing only x
        inputs_embeds = self.wte(x.input_ids)
        position_embeds = self.wpe(x.position_ids)
        x.hidden_states = inputs_embeds + position_embeds
        return x


class MQATransformerBlock(nn.Module):

    def __init__(self, config: TransformerConfig, layer_index):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.attn = MQAAttention(config.hidden_dim, config.n_heads, config.head_dim, config.resid_pdrop,
                                 config.attn_pdrop, layer_index)
        self.ln_2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config.hidden_dim, config.intermediate_dim, config.resid_pdrop,
                       config.activation_function, layer_index)
        self.layer_index = layer_index

    def clear_kv_cache(self):
        self.attn.clear_kv_cache()

    def forward(self, x: TransformerData):
        residual = x.hidden_states
        x.hidden_states = self.ln_1(x.hidden_states)
        x = self.attn(x)
        x.hidden_states = x.hidden_states + residual
        residual = x.hidden_states
        x.hidden_states = self.ln_2(x.hidden_states)
        x = self.mlp(x)
        x.hidden_states = x.hidden_states + residual
        return x


class MQAAttention(nn.Module):

    def __init__(self, hidden_dim, n_heads, head_dim, resid_pdrop, attn_pdrop, layer_index):
        super().__init__()

        self.c_attn = nn.Linear(hidden_dim, hidden_dim + 2 * head_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        # constant
        self.mask_value = None
        self.h = hidden_dim
        self.n = n_heads
        self.d = head_dim

        self.layer_index = layer_index

    def _get_mask_value(self, device, dtype):
        # torch.where expects a tensor. We use a cache to avoid recreating it every time.
        if self.mask_value is None or self.mask_value.dtype != dtype or self.mask_value.device != device:
            self.mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)
        return self.mask_value

    def clear_kv_cache(self):
        kv_cache_prefix = "kv_cache_"
        attr_names = dir(self)
        for attr_name in attr_names:
            if attr_name.startswith(kv_cache_prefix):
                delattr(self, attr_name)

    def _attn(self, q, k, v, attention_mask, head_mask):
        # default upcast, scale
        unscale = self.layer_index + 1
        # print("unscale", unscale)
        scale_factor = unscale**-1
        # print("scale factor 0", scale_factor)
        scale_factor /= self.d**0.5
        # print(f"scale_factor: {scale_factor}")

        # q [b, s, h] => [b, s*n, d]
        b, s, h = q.shape
        q = q.reshape(b, s * self.n, self.d)
        # k [b, s, d] => [b, d, s]
        k = k.transpose(1, 2)
        key_length = k.shape[-1]

        # calculate attn_weights
        attn_shape = (b, s, self.n, key_length)
        attn_view = (b, s * self.n, key_length)
        # TODO: why attn_weight first
        attn_weights = torch.empty(attn_view, device=q.device, dtype=q.dtype)
        if q.device.type == "cpu":
            # bug fix for pytorch
            # https://github.com/pytorch/pytorch/pull/96086
            attn_weights.zero_()
            beta = 1
        else:
            beta = 0
        attn_weights = torch.baddbmm(attn_weights, q, k, beta=beta, alpha=scale_factor).view(attn_shape)

        # TODO: upcast softmax
        softmax_dtype = torch.float32
        if attention_mask is None:
            attn_weights = upcast_softmax(attn_weights, unscale, softmax_dtype)
        else:
            mask_value = self._get_mask_value(attn_weights.device, softmax_dtype)
            # print(attn_weights.shape, attention_mask.shape)
            attn_weights = upcast_masked_softmax(attn_weights, attention_mask, mask_value, unscale,
                                                 softmax_dtype)

        attn_weights = self.attn_dropout(attn_weights)

        # TODO: Mask heads if we want to
        if head_mask is not None:
            head_mask = head_mask.transpose(1, 2)
            attn_weights = attn_weights * head_mask

        # print("debug shapes: ", attn_view, b, s, h)
        attn_output = torch.bmm(attn_weights.view(attn_view), v)
        attn_output = attn_output.view(b, s, h)
        return attn_output

    def forward(self, x: TransformerData):
        # get kv cache from generate_id
        kv_cache = None
        kv_cache_name = "kv_cache_" + str(x.generation_id)
        # print(generation_id)
        if x.generation_id is not None:
            kv_cache = getattr(self, kv_cache_name, None)

        # print(kv_cache)
        # in generate: first pass s=prompt_seq_len, others s=1
        # q [b, s, h] = x.h [b, s, h] * wq [h, h]
        # k [b, s, d] = x.h [b, s, h] * wk [h, d]
        # v [b, s, d] = x.h [b, s, h] * wv [h, d]
        qkv = self.c_attn(x.hidden_states)
        q, kv = torch.split(qkv, (self.h, 2 * self.d), dim=2)
        # use cache in default, for generation
        # k [b, s, d], v [b, s, d], s=current_seq_len
        if kv_cache is not None:
            kv = torch.cat([kv_cache, kv], dim=1)
        if x.generation_id is not None:
            self.register_buffer(kv_cache_name, kv, persistent=False)
        k, v = torch.split(kv, (self.d, self.d), dim=2)

        x.hidden_states = self._attn(q, k, v, x.attention_mask, None)
        x.hidden_states = self.c_proj(x.hidden_states)
        x.hidden_states = self.resid_dropout(x.hidden_states)

        # TODO: graceful way to clear buffer
        # if x.clear_buffer:
        #     delattr(self, kv_cache_name)

        return x


class MLP(nn.Module):

    def __init__(self, hidden_dim, intermediate_dim, resid_pdrop, activation_function, layer_index):
        super().__init__()
        self.c_fc = nn.Linear(hidden_dim, intermediate_dim)
        self.c_proj = nn.Linear(intermediate_dim, hidden_dim)
        if activation_function == "gelu":
            self.act = GELUActivation()
        else:
            raise NotImplementedError("Only \"gelu\" activation function is available.")
        self.dropout = nn.Dropout(resid_pdrop)

        self.layer_index = layer_index

    def forward(self, x: TransformerData):
        hidden_states = x.hidden_states
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        x.hidden_states = hidden_states
        return x


class MQATransformerPipe(PipelineModule):

    def __init__(self, config: TransformerConfig, topology: PipeDataParallelTopology):
        self.config = config

        self.layer_specs = []

        self.layer_specs.append(LayerSpec(PreprocessPipe, config.n_positions))
        self.layer_specs.append(
            LayerSpec(VocabPositionEmbeddingPipe, config.vocab_size, config.n_positions, config.hidden_dim))

        for i in range(config.n_layers):
            self.layer_specs.append(LayerSpec(MQATransformerBlockPipe, config, i))

        self.layer_specs.append(LayerSpec(LastLayerNormPipe, config.hidden_dim, config.layer_norm_epsilon))
        self.layer_specs.append(LayerSpec(PostprocessPipe, config.hidden_dim, config.vocab_size))

        def compute_loss(output, label):
            return output.loss

        super().__init__(layers=self.layer_specs, loss_fn=compute_loss, topology=topology)

    def load_from_path(self, model_path, device, dtype):
        try:
            state_dict = torch.load(model_path)
        except IsADirectoryError:
            import os
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))

        new_state_dict = {}

        # TODO: only load partial keys from files in the future to save memory
        keys = sorted(self.state_dict().keys())
        # TODO: only for 4 layers starcoder to test pipeline,
        #       support automatic key mapping!
        key_map = dict()

        key_map["1.wte.weight"] = "transformer.wte.weight"
        key_map["1.wpe.weight"] = "transformer.wpe.weight"
        key_map["6.inner.weight"] = "transformer.ln_f.weight"
        key_map["6.inner.bias"] = "transformer.ln_f.bias"
        key_map["7.lm_head.weight"] = "lm_head.weight"

        for i in range(4):
            suffices = [
                "ln_1.weight", "ln_1.bias", "attn.c_attn.weight", "attn.c_attn.bias", "attn.c_proj.weight",
                "attn.c_proj.bias", "ln_2.weight", "ln_2.bias", "mlp.c_fc.weight", "mlp.c_fc.bias",
                "mlp.c_proj.weight", "mlp.c_proj.bias"
            ]
            for suffix in suffices:
                key_map[f"{i+2}.{suffix}"] = f"transformer.h.{i}.{suffix}"

        for key in keys:
            loaded_key = key_map[key]
            new_state_dict[key] = state_dict[loaded_key]

        self.load_state_dict(new_state_dict)
        self.to(dtype=dtype, device=device)


# TODO: make a pipeclass wrapper


class PreprocessPipe(Preprocess):

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, inputs, **kwargs):
        # from inputs to TransformerData
        x = TransformerData.from_tuple(inputs)
        out = super().forward(x)
        return out.to_tuple()


class PostprocessPipe(Postprocess):

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, inputs, **kwargs):
        # from inputs to TransformerData
        x = TransformerData.from_tuple(inputs)
        out = super().forward(x)
        return out.to_tuple()


class VocabPositionEmbeddingPipe(VocabPositionEmbedding):

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, inputs, **kwargs):
        # from inputs to TransformerData
        x = TransformerData.from_tuple(inputs)
        out = super().forward(x)
        return out.to_tuple()


class MQATransformerBlockPipe(MQATransformerBlock):

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, inputs, **kwargs):
        # from inputs to TransformerData
        x = TransformerData.from_tuple(inputs)
        out = super().forward(x)
        return out.to_tuple()


class LastLayerNormPipe(LastLayerNorm):

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, inputs, **kwargs):
        # from inputs to TransformerData
        x = TransformerData.from_tuple(inputs)
        out = super().forward(x)
        return out.to_tuple()


def create_mqa_transformer(model_name_or_path, config, name, device):
    # tokenizer = api.huggingface.load_hf_tokenizer(model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path,
                                                           fast_tokenizer=True,
                                                           padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    module = MQATransformerForCausalLM(config)
    module.load_from_path(model_name_or_path, device, torch.half)
    return api.model.Model(name, module, tokenizer, device)


def create_mqa_transformer_pipe(model_name_or_path, config, topology, device, name):
    # tokenizer = api.huggingface.load_hf_tokenizer(model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path,
                                                           fast_tokenizer=True,
                                                           padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    module = MQATransformerPipe(config, topology)
    module.load_from_path(model_name_or_path, device, torch.half)
    return api.model.Model(name, module, tokenizer, device)


api.model.register_model("mqa_transformer", create_mqa_transformer)
api.model.register_model("mqa_transformer_pipe", create_mqa_transformer_pipe)


@dataclasses.dataclass
class GenerationConfig:
    # generation arguments other than tensors
    min_new_tokens: int = 1
    max_new_tokens: int = 10
    min_tokens: int = None
    max_tokens: int = None
    temperature: float = 1.0
    greedy: bool = True
    top_p: float = 1.0
    top_k: int = 0
    num_samples: int = 1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


def preprocess_input(x: TransformerData, last_token_only):
    # x = TransformerData()
    position_ids = x.raw_attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(x.raw_attention_mask == 0, 1)

    if last_token_only:
        x.input_ids = x.raw_input_ids[:, -1:]
        x.position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        x.input_ids = x.raw_input_ids
        x.position_ids = position_ids

    x.attention_mask = x.raw_attention_mask
    return x


def postprocess_output(
    x: TransformerData,
    unfinished_sequences: torch.Tensor,
    args: GenerationConfig,
):
    logits = x.logits
    orig_input_ids = x.raw_input_ids
    orig_attn_mask = x.raw_attention_mask
    cur_seq_len = orig_input_ids.shape[-1]

    # process logits
    next_token_logits = logits[:, -1, :]
    if cur_seq_len < args.min_tokens:
        next_token_logits = mask_eos_token(next_token_logits, args.eos_token_id)

    if args.greedy:
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        print("greedy next tokens", next_tokens, next_tokens.shape)
    else:
        next_token_logits = next_token_logits.float()
        next_token_logits /= args.temperature
        next_token_logits = top_k_top_p_logits(next_token_logits,
                                               top_k=args.top_k,
                                               top_p=args.top_p,
                                               inplace=True,
                                               ordered=True)
        log_probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(log_probs, num_samples=1)
        next_tokens = next_tokens.reshape(-1)

    if args.eos_token_id is not None:
        if args.pad_token_id is None:
            raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        next_tokens = next_tokens * unfinished_sequences + args.pad_token_id * (1 - unfinished_sequences)
    unfinished_sequences = next_tokens.ne(args.eos_token_id).long() * unfinished_sequences
    x.raw_input_ids = torch.cat([orig_input_ids, next_tokens[:, None]], dim=-1)
    x.raw_attention_mask = torch.cat(
        [orig_attn_mask, orig_attn_mask.new_ones((orig_attn_mask.shape[0], 1))], dim=-1)

    # terminate check
    terminate = (cur_seq_len >= args.max_tokens) or (unfinished_sequences.max() == 0)

    return terminate, unfinished_sequences, x


@torch.no_grad()
def generate(
        model,  # could be a deepspeed pipeline engine 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor = None,
        **kwargs):
    args = GenerationConfig(**kwargs)
    device = input_ids.device
    input_length = input_ids.shape[-1]
    # print(args.min_new_tokens, input_length)
    if args.min_tokens is None:
        args.min_tokens = args.min_new_tokens + input_length
        # print(args.min_tokens)
    if args.max_tokens is None:
        args.max_tokens = args.max_new_tokens + input_length

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=device)
    first_round = True
    x = TransformerData(raw_input_ids=input_ids, raw_attention_mask=attention_mask)

    while True:
        x = preprocess_input(x, last_token_only=not first_round)
        first_round = False
        x = model(x)
        terminate, unfinished_sequences, x = \
                postprocess_output(x, unfinished_sequences, args)

        if terminate:
            break

    return x.raw_input_ids
