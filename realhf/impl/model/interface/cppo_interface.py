import collections
import dataclasses
import functools
import itertools
import time
import tqdm
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist

import realhf.api.core.model_api as model_api
import realhf.base.constants as constants
import realhf.base.logging as logging
import realhf.impl.model.utils.cppo_functional as cppo_functional
from realhf.api.core.data_api import SequenceSample
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.nn.real_llm_generate import concat_prompt_to_generation_output
from realhf.impl.model.utils.functional import (
    apply_logits_mask,
    gather_packed_shifted_log_probs,
    masked_normalization,
)

logger = logging.getLogger("PackedCPPOInterface")


def _cppo_actor_loss_from_model_outputs(
    logits: torch.FloatTensor,  # [tot_seqlen, vocab_size]
    input_: SequenceSample,
    kl_adapter: cppo_functional.KLController,  # const
    eps_clip: float,  # const
    early_stop_imp_ratio: Optional[float],  # const
    early_stop_kl: Optional[float],  # const
) -> Tuple[torch.FloatTensor, Dict]:
    """Loss function for cppo actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    logits_mask = input_.data["packed_logits_mask"]
    packed_input_ids = input_.data["packed_input_ids"]
    cu_seqlens = (
        torch.nn.functional.pad(
            torch.cat(input_.seqlens["packed_input_ids"]).cumsum(0), (1, 0)
        )
        .int()
        .cuda()
    )
    cppo_loss_mask = input_.data["cppo_loss_mask"]
    advantages = input_.data["advantages"]
    old_logp = input_.data["old_logp"]
    kl_rewards = input_.data["kl_rewards"]

    if logits_mask is not None:
        apply_logits_mask(logits, logits_mask)

    n_tokens = cppo_loss_mask.count_nonzero()
    logprobs = gather_packed_shifted_log_probs(
        logits, cu_seqlens, packed_input_ids
    ).float()
    loss, cppo_stat = cppo_functional.actor_loss_fn(
        logprobs=logprobs,
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        loss_mask=cppo_loss_mask,
    )

    importance_weight = cppo_stat["importance_weight"].float() * n_tokens
    clip_ratio = cppo_stat["clip_ratio"].float() * n_tokens
    approx_kl = cppo_stat["approx_kl"].float() * n_tokens

    # Logging and early stopping according to KL (logp vs ref) or importance ratio (new logp vs old logp).
    mean_ref_kl = (kl_rewards.detach().float() * cppo_loss_mask).sum()
    logging_loss = torch.where(cppo_loss_mask, loss.detach().float(), 0.0).sum()
    dist.all_reduce(n_tokens, group=constants.data_parallel_group())
    dist.all_reduce(mean_ref_kl, group=constants.data_parallel_group())
    dist.all_reduce(importance_weight, group=constants.data_parallel_group())
    dist.all_reduce(clip_ratio, group=constants.data_parallel_group())
    dist.all_reduce(approx_kl, group=constants.data_parallel_group())
    dist.all_reduce(logging_loss, group=constants.data_parallel_group())

    # Early stopping.
    kl_adapter.update(mean_ref_kl / n_tokens, n_steps=cu_seqlens.shape[0] - 1)
    _imp = importance_weight / n_tokens
    _kl = approx_kl / n_tokens
    if early_stop_imp_ratio is not None and _imp > early_stop_imp_ratio:
        logger.warning(
            f"Current importance ratio {_imp.item():.4f} is larger "
            f"than early stop threshold {early_stop_imp_ratio}. Abandon this minibatch."
        )
        loss = loss * 0.0
    if early_stop_kl is not None and _kl > early_stop_kl:
        logger.warning(
            f"Current approximate KL divergence {_kl.item():.4f} is larger "
            f"than early stop threshold {early_stop_kl}. Abort actor update."
        )
        loss = loss * 0.0

    stats = dict(
        cppo_approx_kl=approx_kl,
        actor_loss=logging_loss,
        actor_clip_ratio=clip_ratio,
        importance_weight=importance_weight,
    )

    return loss, stats


@dataclasses.dataclass
class CPPOActorInterface(model_api.ModelInterface):
    n_minibatches: int = 4

    generation_config: model_api.GenerationHyperparameters = dataclasses.field(
        default_factory=model_api.GenerationHyperparameters
    )

    kl_ctl: float = 0.1

    adv_norm: bool = True
    discount: float = 1.0
    gae_lambda: float = 1.0

    eps_clip: float = 0.2
    value_eps_clip: float = 0.2
    max_reward_clip: float = 5.0

    early_stop_kl: Optional[float] = None  # e.g. 0.1
    early_stop_imp_ratio: Optional[float] = None  # e.g., 10.0

    adaptive_kl_ctl: bool = False
    adaptive_kl_target: Optional[float] = 6
    adaptive_kl_horizon: Optional[float] = 10000

    adaptive_cost_ctl: bool = False
    

    enable_save: bool = True
    force_no_logits_mask: bool = False

    value_norm: bool = False
    value_norm_type: str = dataclasses.field(
        metadata={"choices": ["exp", "ma"]}, default="exp"
    )
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5

    def __post_init__(self):
        if self.adaptive_kl_ctl:
            assert self.adaptive_kl_target is not None
            assert self.adaptive_kl_horizon is not None
            self.kl_adapter = cppo_functional.AdaptiveKLController(
                self.kl_ctl, self.adaptive_kl_target, self.adaptive_kl_horizon
            )
        else:
            self.kl_adapter = cppo_functional.FixedKLController(self.kl_ctl)
        if self.value_norm:
            from realhf.impl.model.modules import (
                ExponentialRunningMeanStd,
                MovingAverageRunningMeanStd,
            )

            if self.value_norm_type == "exp":
                self.rms = ExponentialRunningMeanStd(
                    beta=self.value_norm_beta, epsilon=self.value_norm_eps
                )
            elif self.value_norm_type == "ma":
                self.rms = MovingAverageRunningMeanStd()
            else:
                raise ValueError(f"Unknown value_norm_type {self.value_norm_type}")
        self.kl_ctl = None

    def save(self, model: model_api.Model, save_dir: str):
        if not self.enable_save:
            return
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        module.save_to_hf(
            tokenizer=model.tokenizer,
            save_dir=save_dir,
        )

    @torch.no_grad()
    def generate(
        self, model: model_api.Model, input_: SequenceSample
    ) -> SequenceSample:
        module = model.module

        module.eval()

        # Remap the key `packed_prompts` to `packed_input_ids`,
        # because the pipe runner only recognizes `packed_input_ids`.
        x = SequenceSample.from_default(
            ids=input_.ids,
            seqlens=input_.seqlens["packed_prompts"],
            data=dict(packed_input_ids=input_.data["packed_prompts"]),
        )

        res = module.generate(
            input_=x,
            tokenizer=model.tokenizer,
            gconfig=self.generation_config,
        )
        if res is None:
            return None

        gen_tokens, logprobs, logits_mask, *_ = res

        pad_token_id = model.tokenizer.pad_token_id
        eos_token_id = model.tokenizer.eos_token_id
        seq_no_eos_mask = (gen_tokens[:, -1] != eos_token_id).logical_and(
            gen_tokens[:, -1] != pad_token_id
        )
        # We also want gen_lengths to include the eos token, where the reward model outputs a score for this sequence.
        gen_lengths = (gen_tokens != pad_token_id).logical_and(
            gen_tokens != eos_token_id
        ).sum(dim=-1) + 1
        gen_lengths = gen_lengths.clip(max=gen_tokens.shape[-1])

        (
            packed_input_ids,
            packed_logprobs,
            packed_logits_mask,
            seq_lengths,
            prompt_mask,
        ) = concat_prompt_to_generation_output(
            packed_prompts=input_.data["packed_prompts"],
            prompt_lengths=torch.cat(input_.seqlens["packed_prompts"]).to(model.device),
            gen_tokens=gen_tokens,
            logprobs=logprobs,
            logits_mask=logits_mask,
            gen_lengths=gen_lengths,
        )

        seqlens = [
            torch.tensor([s], dtype=torch.int32)
            for s in seq_lengths.cpu().numpy().tolist()
        ]
        res = SequenceSample.from_default(
            ids=input_.ids,
            seqlens=seqlens,
            data=dict(
                seq_no_eos_mask=seq_no_eos_mask,
                packed_input_ids=packed_input_ids,
                packed_logprobs=packed_logprobs,
                packed_logits_mask=(
                    packed_logits_mask.bool()
                    if not self.force_no_logits_mask and packed_logits_mask is not None
                    else None
                ),
                prompt_mask=prompt_mask,
            ),
        )
        return res

    @torch.no_grad()
    def inference(
        self, model: model_api.Model, input_: SequenceSample
    ) -> SequenceSample:
        module = model.module
        module.eval()

        logits = module.forward(input_=input_)
        if logits is None:
            return None

        logits /= self.generation_config.temperature
        if (
            "packed_logits_mask" in input_.data
            and input_.data["packed_logits_mask"] is not None
        ):
            apply_logits_mask(logits, input_.data["packed_logits_mask"])
        input_lens = torch.cat(input_.seqlens["packed_input_ids"])
        cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
        logprobs = gather_packed_shifted_log_probs(
            logits, cu_seqlens, input_.data["packed_input_ids"]
        )
        res = SequenceSample.from_default(
            ids=input_.ids,
            seqlens=input_.seqlens["packed_input_ids"],
            data=dict(packed_ref_logprobs=logprobs),
        )
        return res

    def train_step(self, model: model_api.Model, input_: SequenceSample) -> Dict:
        module = model.module
        # We call module.eval() because dropout causes the computation of incorrect of log probs.
        module.eval()

        old_logp: torch.FloatTensor = input_.data["packed_logprobs"].float()
        ref_logp: torch.FloatTensor = input_.data["packed_ref_logprobs"].float()
        prompt_mask = input_.data["prompt_mask"]
        input_lens = torch.cat(input_.seqlens["packed_input_ids"]).cuda()
        cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
        reward_score = input_.data["rewards"].float()
        values = input_.data["values"].float()
        seq_no_eos_mask = input_.data["seq_no_eos_mask"]

        if self.value_norm:
            denormalized_values = self.rms.denormalize(values)
        else:
            denormalized_values = values

        for i in range(seq_no_eos_mask.shape[0]):
            if not seq_no_eos_mask[i]:
                # Set value at the EOS token to be zero.
                denormalized_values[cu_seqlens[i + 1] - 1] = 0.0
                values[cu_seqlens[i + 1] - 1] = 0.0

        # Shift the loss mask by one token for each packed sequences.
        short1cu_seqlens = cu_seqlens.clone()
        short1cu_seqlens[1:] -= torch.ones_like(cu_seqlens[1:]).cumsum(0)
        loss_mask = prompt_mask.logical_not()
        shift_one_indices = torch.cat(
            [
                torch.arange(
                    cu_seqlens[i] + 1,
                    cu_seqlens[i + 1],
                    dtype=torch.long,
                    device=cu_seqlens.device,
                )
                for i in range(cu_seqlens.shape[0] - 1)
            ]
        )
        loss_mask = loss_mask[shift_one_indices]

        # Apply the mask to log probabilities.
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute rewards and GAEs.
        kl_rewards, rewards = cppo_functional.get_packed_rewards(
            kl_ctl=self.kl_adapter.value,
            clip_reward_value=self.max_reward_clip,
            log_probs=old_logp,
            ref_log_probs=ref_logp,
            reward_score=reward_score,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )
        advantages, returns = cppo_functional.get_packed_advantages_and_returns(
            gamma=self.discount,
            lam=self.gae_lambda,
            values=denormalized_values,
            rewards=rewards,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )

        # Optionally perform normalization.
        if self.value_norm:
            self.rms.update(returns, mask=loss_mask)
        if self.adv_norm:
            advantages = masked_normalization(advantages, loss_mask)

        # Prepare data to be splitted into mini-batches.
        input_ = SequenceSample.from_default(
            ids=input_.ids,
            data=dict(
                advantages=advantages,
                old_logp=old_logp,
                cppo_loss_mask=loss_mask,
                packed_input_ids=input_.data["packed_input_ids"],
                kl_rewards=kl_rewards,
                packed_logits_mask=(
                    input_.data["packed_logits_mask"]
                    if "packed_logits_mask" in input_.data
                    else None
                ),
            ),
            seqlens=input_.seqlens["packed_input_ids"],
        )
        # NOTE: We cannot randomly shuffle data here because
        # data must have the same shape across different pipeline stages.
        datas = input_.split(
            self.n_minibatches,
            min_size=constants.pipe_parallel_world_size() * 2,
        )

        ### Logging code starts. ###
        _n_seqs = torch.tensor(
            [reward_score.shape[0]], dtype=torch.float32, device=model.device
        )
        _n_tokens = loss_mask.count_nonzero()
        task_reward = reward_score.sum()
        _advantages = advantages.sum()
        _kl_rewards = (kl_rewards * loss_mask).sum()
        prompt_len = prompt_mask.count_nonzero().float()
        seq_len = input_lens.float().sum()
        dist.all_reduce(_n_seqs, group=constants.data_parallel_group())
        dist.all_reduce(task_reward, group=constants.data_parallel_group())
        dist.all_reduce(_advantages, group=constants.data_parallel_group())
        dist.all_reduce(prompt_len, group=constants.data_parallel_group())
        dist.all_reduce(seq_len, group=constants.data_parallel_group())
        dist.all_reduce(_n_tokens, group=constants.data_parallel_group())
        dist.all_reduce(_kl_rewards, group=constants.data_parallel_group())
        global_stats = dict(
            task_reward=float(task_reward / _n_seqs),
            kl_reward=float(_kl_rewards / _n_tokens),
            advantage=float(_advantages / _n_tokens),
            avg_seq_len=float(seq_len / _n_seqs),
            avg_prompt_len=float(prompt_len / _n_seqs),
            n_tokens=int(_n_tokens),
            n_seqs=int(_n_seqs),
        )

        if input_.data["packed_logits_mask"] is not None:
            n_masked_vocabs = input_.data["packed_logits_mask"].count_nonzero()
            total_vocabs = torch.tensor(
                input_.data["packed_logits_mask"].numel(),
                dtype=torch.long,
                device=model.device,
            )
            dist.all_reduce(n_masked_vocabs, group=constants.data_parallel_group())
            dist.all_reduce(total_vocabs, group=constants.data_parallel_group())
            global_stats["valid_vocab_ratio"] = float(
                (total_vocabs - n_masked_vocabs) / total_vocabs
            )
        ### Logging code ends. ###

        # Run mini-batched CPPO training!
        train_stats = collections.defaultdict(lambda: 0)
        for data in datas:
            stats = module.train_batch(
                input_=data,
                version_steps=model.version.global_step,
                loss_fn=functools.partial(
                    _cppo_actor_loss_from_model_outputs,
                    kl_adapter=self.kl_adapter,
                    eps_clip=self.eps_clip,
                    early_stop_imp_ratio=self.early_stop_imp_ratio,
                    early_stop_kl=self.early_stop_kl,
                ),
            )

            if stats:
                for k, v in stats.items():
                    train_stats[k] += v
        cur_epoch = model.version.epoch
        model.inc_version()

        if train_stats:
            train_stats = dict(
                cppo_approx_kl=float(train_stats["cppo_approx_kl"] / _n_tokens),
                actor_loss=float(train_stats["actor_loss"] / _n_tokens),
                actor_clip_ratio=float(train_stats["actor_clip_ratio"] / _n_tokens),
                importance_weight=float(train_stats["importance_weight"] / _n_tokens),
            )
            train_stats = dict(**train_stats, **global_stats)

        return dict(train_stats)


def _cppo_critic_loss_from_model_outputs(
    new_values: torch.FloatTensor,
    input_: SequenceSample,
    value_eps_clip: float,
    kl_adapter: cppo_functional.KLController,
    rms=None,
) -> Tuple[torch.FloatTensor, Dict]:

    cu_seqlens = (
        torch.nn.functional.pad(
            torch.cat(input_.seqlens["packed_input_ids"]).cumsum(0), (1, 0)
        )
        .int()
        .cuda()
    )
    cppo_loss_mask = input_.data["cppo_loss_mask"]
    returns = input_.data["returns"]
    values = input_.data["values"]
    kl_rewards = input_.data["kl_rewards"]

    leave_one_indices = torch.cat(
        [
            torch.arange(
                cu_seqlens[i],
                cu_seqlens[i + 1] - 1,
                dtype=torch.long,
                device=cu_seqlens.device,
            )
            for i in range(cu_seqlens.shape[0] - 1)
        ]
    )
    new_values = new_values[leave_one_indices].squeeze(-1).float()
    values = values[leave_one_indices].squeeze(-1).float()

    loss, loss_stat = cppo_functional.critic_loss_fn(
        value=new_values,
        old_value=values,
        target_value=returns,
        value_eps_clip=value_eps_clip,
        loss_mask=cppo_loss_mask,
    )

    if rms is not None:
        denormalized_values = rms.denormalize(new_values)
    else:
        denormalized_values = new_values

    # Logging.
    n_tokens = cppo_loss_mask.count_nonzero()
    mean_ref_kl = (kl_rewards.detach().float() * cppo_loss_mask).sum()
    logging_loss = loss.detach().float() * n_tokens
    clip_ratio = loss_stat["clip_ratio"].float() * n_tokens
    denormalized_values = (
        torch.where(cppo_loss_mask, denormalized_values, 0.0).sum().detach().float()
    )
    dist.all_reduce(n_tokens, group=constants.data_parallel_group())
    dist.all_reduce(mean_ref_kl, group=constants.data_parallel_group())
    dist.all_reduce(logging_loss, group=constants.data_parallel_group())
    dist.all_reduce(clip_ratio, group=constants.data_parallel_group())
    dist.all_reduce(denormalized_values, group=constants.data_parallel_group())

    # Update KL coefficient to be consistent with actor.
    kl_adapter.update(mean_ref_kl, n_steps=cu_seqlens.shape[0] - 1)

    return loss, dict(
        value_loss=logging_loss,
        value_clip_ratio=clip_ratio,
        denormalized_values=denormalized_values,
    )


@dataclasses.dataclass
class CPPOCriticInterface(model_api.ModelInterface):
    n_minibatches: int = 4
    enable_save: bool = True
    kl_ctl: float = 0.1
    discount: float = 1.0
    gae_lambda: float = 0.95
    value_eps_clip: float = 0.2
    max_reward_clip: float = 5.0
    adaptive_kl_ctl: bool = False
    adaptive_kl_target: Optional[float] = 6
    adaptive_kl_horizon: Optional[float] = 10000
    value_norm: bool = False
    value_norm_type: str = dataclasses.field(
        metadata={"choices": ["exp", "ma"]}, default="exp"
    )
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5

    def __post_init__(self):
        if self.adaptive_kl_ctl:
            assert self.adaptive_kl_target is not None
            assert self.adaptive_kl_horizon is not None
            self.kl_adapter = cppo_functional.AdaptiveKLController(
                self.kl_ctl, self.adaptive_kl_target, self.adaptive_kl_horizon
            )
        else:
            self.kl_adapter = cppo_functional.FixedKLController(self.kl_ctl)
        if self.value_norm:
            from realhf.impl.model.modules import (
                ExponentialRunningMeanStd,
                MovingAverageRunningMeanStd,
            )

            if self.value_norm_type == "exp":
                self.rms = ExponentialRunningMeanStd(
                    beta=self.value_norm_beta, epsilon=self.value_norm_eps
                )
            elif self.value_norm_type == "ma":
                self.rms = MovingAverageRunningMeanStd()
            else:
                raise ValueError(f"Unknown value_norm_type {self.value_norm_type}")
        self.kl_ctl = None

    def save(self, model: model_api.Model, save_dir: str):
        if not self.enable_save:
            return
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        module.save_to_hf(
            tokenizer=model.tokenizer,
            save_dir=save_dir,
        )

    @torch.no_grad()
    def inference(
        self, model: model_api.Model, input_: SequenceSample
    ) -> SequenceSample:
        module = model.module
        module.eval()

        scores = module.forward(input_=input_)
        if scores is None:
            return None
        scores = scores.squeeze(-1)
        res = SequenceSample.from_default(
            ids=input_.ids,
            data=dict(values=scores),
            seqlens=input_.seqlens["packed_input_ids"],
        )
        return res

    def train_step(self, model: model_api.Model, input_: SequenceSample) -> Dict:
        module = model.module
        tokenizer = model.tokenizer
        # We call module.eval() because dropout causes the computation of incorrect of log probs.
        module.eval()

        old_logp: torch.FloatTensor = input_.data["packed_logprobs"].float()
        ref_logp: torch.FloatTensor = input_.data["packed_ref_logprobs"].float()
        prompt_mask = input_.data["prompt_mask"]
        input_lens = torch.cat(input_.seqlens["packed_input_ids"]).cuda()
        cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
        reward_score = input_.data["rewards"].float()
        values = input_.data["values"].float()
        seq_no_eos_mask = input_.data["seq_no_eos_mask"]

        if self.value_norm:
            denormalized_values = self.rms.denormalize(values)
        else:
            denormalized_values = values

        for i in range(seq_no_eos_mask.shape[0]):
            if not seq_no_eos_mask[i]:
                # Set value at the EOS token to be zero.
                denormalized_values[cu_seqlens[i + 1] - 1] = 0.0
                values[cu_seqlens[i + 1] - 1] = 0.0

        # Shift the loss mask by one token for each packed sequences.
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        short1cu_seqlens = cu_seqlens.clone()
        short1cu_seqlens[1:] -= torch.ones_like(cu_seqlens[1:]).cumsum(0)
        loss_mask = prompt_mask.logical_not()
        shift_one_indices = torch.cat(
            [
                torch.arange(
                    cu_seqlens[i] + 1,
                    cu_seqlens[i + 1],
                    dtype=torch.long,
                    device=cu_seqlens.device,
                )
                for i in range(cu_seqlens.shape[0] - 1)
            ]
        )
        loss_mask = loss_mask[shift_one_indices]

        # Apply the mask to log probabilities.
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute rewards and GAEs.
        kl_rewards, rewards = cppo_functional.get_packed_rewards(
            kl_ctl=self.kl_adapter.value,
            clip_reward_value=self.max_reward_clip,
            log_probs=old_logp,
            ref_log_probs=ref_logp,
            reward_score=reward_score,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )
        _, returns = cppo_functional.get_packed_advantages_and_returns(
            gamma=self.discount,
            lam=self.gae_lambda,
            values=denormalized_values,
            rewards=rewards,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )

        # Optionally perform normalization.
        if self.value_norm:
            self.rms.update(returns, mask=loss_mask)
            normalized_returns = self.rms.normalize(returns)
        else:
            normalized_returns = returns

        # Prepare data to be splitted into mini-batches.
        input_ = SequenceSample.from_default(
            ids=input_.ids,
            data=dict(
                returns=normalized_returns,
                values=values,
                cppo_loss_mask=loss_mask,
                packed_input_ids=input_.data["packed_input_ids"],
                kl_rewards=kl_rewards,
            ),
            seqlens=input_.seqlens["packed_input_ids"],
        )
        # NOTE: We cannot randomly shuffle data here because
        # data must have the same shape across different pipeline stages.
        datas = input_.split(
            self.n_minibatches,
            min_size=constants.pipe_parallel_world_size() * 2,
        )

        # Logging.
        returns = torch.where(loss_mask, returns, 0.0).sum()
        n_tokens = loss_mask.count_nonzero()
        dist.all_reduce(returns, group=constants.data_parallel_group())
        dist.all_reduce(n_tokens, group=constants.data_parallel_group())
        global_stats = dict(returns=float(returns), n_tokens=int(n_tokens))

        # Run mini-batched CPPO training!
        train_stats = collections.defaultdict(lambda: 0)
        for data in datas:

            stats = module.train_batch(
                input_=data,
                version_steps=model.version.global_step,
                loss_fn=functools.partial(
                    _cppo_critic_loss_from_model_outputs,
                    value_eps_clip=self.value_eps_clip,
                    kl_adapter=self.kl_adapter,
                    rms=None if not self.value_norm else self.rms,
                ),
            )

            if stats:
                for k, v in stats.items():
                    train_stats[k] += v

        cur_epoch = model.version.epoch
        model.inc_version()

        if train_stats:
            train_stats = dict(
                value_loss=float(train_stats["value_loss"] / n_tokens),
                value_clip_ratio=float(train_stats["value_clip_ratio"] / n_tokens),
                denormalized_values=float(
                    train_stats["denormalized_values"] / n_tokens
                ),
                returns=global_stats["returns"] / int(n_tokens),
                n_tokens=int(n_tokens),
            )

        return dict(train_stats)

def flatten_list(l):
    return list(itertools.chain(*l))


def _cppo_rw_loss_from_model_outputs(
    scores: torch.FloatTensor,
    input_: SequenceSample,
):
    # Normalize pairs of each prompt with the group factor,
    # which is the reciprocal of the number of pairs in the group.
    group_sizes = [len(x) // 2 for x in input_.seqlens["packed_input_ids"]]
    assert all([x >= 1 for x in group_sizes])
    group_factor = torch.tensor(
        flatten_list([[1 / g for _ in range(g)] for g in group_sizes]),
        device=scores.device,
    )

    input_lens = torch.cat(input_.seqlens["packed_input_ids"])

    assert scores.shape[0] == input_lens.sum(), (scores.shape, input_lens.sum())
    scores = scores[input_lens.cumsum(0) - 1].view(-1, 2).float()
    loss = -(
        torch.nn.functional.logsigmoid(scores[:, 0] - scores[:, 1]) * group_factor
    ).sum()

    # Logging.
    correct_predictions = (scores[:, 0] > scores[:, 1]).count_nonzero().detach().float()
    total_predictions = torch.tensor(
        scores.shape[0], dtype=torch.float32, device=scores.device
    )
    dist.all_reduce(
        correct_predictions,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )
    dist.all_reduce(
        total_predictions,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )
    pos_score_sum = scores[:, 0].sum().detach()
    max_pos_score = scores[:, 0].max(dim=0).values
    neg_score_sum = scores[:, 1].sum().detach()
    min_neg_score = scores[:, 1].min(dim=0).values
    dist.all_reduce(
        pos_score_sum,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )
    dist.all_reduce(
        neg_score_sum,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )
    loss_logging = loss.detach()
    dist.all_reduce(
        loss_logging,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )
    dist.all_reduce(
        max_pos_score,
        op=dist.ReduceOp.MAX,
        group=constants.data_parallel_group(),
    )
    dist.all_reduce(
        min_neg_score,
        op=dist.ReduceOp.MIN,
        group=constants.data_parallel_group(),
    )
    return loss, dict(
        loss=loss_logging,
        correct_predictions=correct_predictions,
        total_predictions=total_predictions,
        pos_score=pos_score_sum,
        neg_score=neg_score_sum,
        max_pos_score=max_pos_score,
        min_neg_score=min_neg_score,
    )


@dataclasses.dataclass
class CPPORewardInterface(model_api.ModelInterface):
    enable_save: bool = True

    output_scaling: float = 1.0
    output_bias: float = 0.0

    # training log
    train_total_predictions: int = 0
    train_total_correct_predictions: int = 0

    @torch.no_grad()
    def inference(self, model: model_api.Model, data: SequenceSample) -> SequenceSample:
        module = model.module
        module.eval()

        _ = module.forward(input_=data)
    
        cu_intput_seqlens = torch.nn.functional.pad(torch.cat(data.seqlens["packed_input_ids"]).cumsum(0), (1, 0)).int() # [bs + 1]
        cu_target_seqlens = torch.nn.functional.pad(torch.cat(data.seqlens["packed_targets"]).cumsum(0), (1, 0)).int()
        
        input_ids = [data.data["packed_input_ids"][start:end] for start, end in zip(cu_intput_seqlens[:-1], cu_intput_seqlens[1:])]
        target_ids = [data.data["packed_targets"][start:end] for start, end in zip(cu_target_seqlens[:-1], cu_target_seqlens[1:])]
        
        scores = torch.tensor([cppo_functional.is_substring(x, y[2:]) for x, y in zip(input_ids, target_ids)], dtype=torch.float32, device=model.device)
        scores = (scores - self.output_bias) * self.output_scaling
        ###################### logging ######################
        # input_ids = [packed_input_ids[start:end] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
        # seq_strs = model.tokenizer.batch_decode(input_ids,
        #                                         clean_up_tokenization_spaces=False,
        #                                         skip_special_tokens=True)
        # for seq_str, score in zip(seq_strs, scores):
        #     logger.info(
        #         f"reward is {colorama.Fore.RED}{score.item()}{colorama.Style.RESET_ALL}, "
        #         f"sequence is: {colorama.Fore.YELLOW + colorama.Style.DIM}{seq_str}{colorama.Style.RESET_ALL}"
        #     )
        #####################################################

        res = SequenceSample(
            keys=["rewards"],
            trailing_shapes=dict(rewards=()),
            dtypes=dict(rewards=torch.float32),
            ids=data.ids,
            seqlens=dict(
                rewards=[
                    torch.tensor([1 for _ in range(len(x))], dtype=torch.int32)
                    for x in data.seqlens["packed_input_ids"]
                ]
            ),
            data=dict(rewards=scores),
        )
        return res

    def train_step(
        self, model: model_api.Model, data: SequenceSample
    ) -> SequenceSample:
        module = model.module
        module.train()

        stats = module.train_batch(
            input_=data,
            loss_fn=_cppo_rw_loss_from_model_outputs,
            version_steps=model.version.global_step,
        )

        res = {}
        if stats:
            if constants.pipe_parallel_world_size() > 1:
                stats["max_pos_score"] /= constants.pipe_parallel_world_size() * 2
                stats["min_neg_score"] /= constants.pipe_parallel_world_size() * 2
            self.train_total_predictions += int(stats["total_predictions"])
            self.train_total_correct_predictions += int(stats["correct_predictions"])
            res = dict(
                loss=float(stats["loss"] / stats["total_predictions"]),
                epoch_acc=self.train_total_correct_predictions
                / self.train_total_predictions,
                batch_acc=float(
                    stats["correct_predictions"] / stats["total_predictions"]
                ),
                avg_pos_score=float(stats["pos_score"] / stats["total_predictions"]),
                avg_neg_score=float(stats["neg_score"] / stats["total_predictions"]),
                total_predictions=int(stats["total_predictions"]),
                correct_predictions=int(stats["correct_predictions"]),
                max_pos_score=float(stats["max_pos_score"]),
                min_neg_score=float(stats["min_neg_score"]),
            )

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            self.train_total_predictions = self.train_total_correct_predictions = 0

        return res

    def save(self, model: model_api.Model, save_dir: str):
        if not self.enable_save:
            return
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        module.save_to_hf(
            tokenizer=model.tokenizer,
            save_dir=save_dir,
        )

    @torch.no_grad()
    def evaluate(
        self,
        model_: model_api.Model,
        eval_dataloader: torch.utils.data.DataLoader,
    ) -> Dict:
        model = model_.module

        model.eval()
        total_predictions = correct_predictions = 0
        losses = 0
        pos_score = neg_score = 0
        max_pos_score = -float("inf")
        min_neg_score = float("inf")

        for step, data in enumerate(tqdm.tqdm(eval_dataloader)):
            data: SequenceSample
            stats = model.eval_batch(
                input_=data.cuda(),
                loss_fn=_cppo_rw_loss_from_model_outputs,
            )

            if stats:
                losses += stats["loss"].item()
                correct_predictions += stats["correct_predictions"].item()
                total_predictions += stats["total_predictions"].item()
                pos_score += stats["pos_score"].item()
                neg_score += stats["neg_score"].item()
                max_pos_score = max(max_pos_score, stats["max_pos_score"].item())
                min_neg_score = min(min_neg_score, stats["min_neg_score"].item())

        if total_predictions > 0:
            return dict(
                loss=float(losses / total_predictions),
                acc=correct_predictions / total_predictions,
                pos_score=float(pos_score / total_predictions),
                neg_score=float(neg_score / total_predictions),
                correct_predictions=int(correct_predictions),
                total_predictions=int(total_predictions),
                max_pos_score=float(max_pos_score),
                min_neg_score=float(min_neg_score),
            )
        return dict()


model_api.register_interface("cppo_rw", CPPORewardInterface)
model_api.register_interface("cppo_actor", CPPOActorInterface)
model_api.register_interface("cppo_critic", CPPOCriticInterface)
