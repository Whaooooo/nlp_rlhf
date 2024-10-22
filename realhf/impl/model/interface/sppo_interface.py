import collections
import dataclasses
import functools
import itertools
import time
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist

import realhf.api.core.model_api as model_api
import realhf.base.constants as constants
import realhf.base.logging as logging
import realhf.impl.model.utils.ppo_functional as ppo_functional
from realhf.api.core.data_api import SequenceSample
from realhf.base.datapack import flat2d
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.nn.real_llm_generate import concat_prompt_to_generation_output
from realhf.impl.model.utils.functional import (
    apply_logits_mask,
    gather_packed_shifted_log_probs,
    masked_normalization,
)

logger = logging.getLogger("PackedSPPOInterface")

def sft_loss_fn(
    logprobs, prompt_mask, cu_seqlens, device,
):
    shift_one_indices = torch.cat([
        torch.arange(cu_seqlens[i] + 1, cu_seqlens[i + 1], dtype=torch.long, device=cu_seqlens.device)
        for i in range(cu_seqlens.shape[0] - 1)
    ])
    prompt_mask = prompt_mask[shift_one_indices]
    logprobs = torch.where(prompt_mask, 0, logprobs)

    loss_sum = -logprobs.sum()

    with torch.no_grad():
        seqlogp = torch.zeros(
            cu_seqlens.shape[0] - 1, device=device, dtype=torch.float64
        )
        for i in range(cu_seqlens.shape[0] - 1):
            m = prompt_mask[cu_seqlens[i] - i : cu_seqlens[i + 1] - i - 1]
            logp = logprobs[cu_seqlens[i] - i : cu_seqlens[i + 1] - i - 1]
            assert cu_seqlens[i + 1] - i - 1 <= logprobs.shape[0], (
                cu_seqlens,
                logprobs.shape,
            )
            seqlogp[i] = torch.where(m, 0.0, logp).sum() / (
                m.numel() - m.count_nonzero()
            ) if m.numel() > m.count_nonzero() else 0.0
    logging_ppl = (-seqlogp).exp().sum()
    token_denorm = prompt_mask.numel() - prompt_mask.count_nonzero()

    # Logging loss and perplexity.
    logging_loss = loss_sum.detach().clone()
    dist.all_reduce(
        logging_ppl, op=dist.ReduceOp.SUM, group=constants.data_parallel_group()
    )
    dist.all_reduce(
        logging_loss,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )

    loss = loss_sum / token_denorm
    return loss, {
        "loss": logging_loss,
        "ppl": logging_ppl,
    }

def _sppo_actor_loss_from_model_outputs(
    logits: torch.FloatTensor,  # [tot_seqlen, vocab_size]
    input_: SequenceSample,
    kl_adapter: ppo_functional.KLController,  # const
    eps_clip: float,  # const
    early_stop_imp_ratio: Optional[float],  # const
    early_stop_kl: Optional[float],  # const
    sft_loss_ratio: float,  # const
) -> Tuple[torch.FloatTensor, Dict]:
    """Loss function for sppo actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    logits_mask = input_.data["packed_logits_mask"]
    packed_input_ids = input_.data["packed_input_ids"]
    cu_seqlens = (
        torch.nn.functional.pad(
            torch.tensor(flat2d(input_.seqlens["packed_input_ids"])).cumsum(0),
            (1, 0),
        )
        .int()
        .cuda()
    )
    sppo_loss_mask = input_.data["sppo_loss_mask"]
    advantages = input_.data["advantages"]
    old_logp = input_.data["old_logp"]
    kl_rewards = input_.data["kl_rewards"]

    if logits_mask is not None:
        # Ensure logits_mask only changes along the 0th dimension, while keeping the 1st dimension unchanged
        padded_mask = torch.zeros(logits.size(0), logits_mask.size(1), device=logits.device, dtype=torch.bool)
        
        # Copy the logits_mask into padded_mask
        padded_mask[:logits_mask.size(0), :] = logits_mask  # Copy over the mask for the first dimension

        # Now call apply_logits_mask with the padded mask
        apply_logits_mask(logits, padded_mask)


    n_tokens = sppo_loss_mask.count_nonzero()
    logprobs = gather_packed_shifted_log_probs(
        logits, cu_seqlens, packed_input_ids
    ).float()

    middle_index = (len(cu_seqlens) - 1) // 2
    middle_len = cu_seqlens[middle_index] - middle_index
    half_cu_len = cu_seqlens[middle_index:] - cu_seqlens[middle_index]

    loss, sppo_stat = ppo_functional.actor_loss_fn(
        logprobs=logprobs[:middle_len],
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        loss_mask=sppo_loss_mask,
    )

    sft_loss, sft_log= sft_loss_fn(
        logprobs=logprobs[middle_len:],
        prompt_mask=input_.data["ref_prompt_mask"],
        cu_seqlens=half_cu_len,
        device=logits.device,
    )

    importance_weight = sppo_stat["importance_weight"].float() * n_tokens
    clip_ratio = sppo_stat["clip_ratio"].float() * n_tokens
    approx_kl = sppo_stat["approx_kl"].float() * n_tokens

    # Logging and early stopping according to KL (logp vs ref) or importance ratio (new logp vs old logp).
    mean_ref_kl = (kl_rewards.detach().float() * sppo_loss_mask).sum()
    logging_loss = torch.where(sppo_loss_mask, loss.detach().float(), 0.0).sum()
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
        sppo_approx_kl=approx_kl,
        actor_loss=logging_loss,
        sft_loss=sft_log["loss"] * sft_loss_ratio,
        sft_ppl=sft_log["ppl"],
        actor_clip_ratio=clip_ratio,
        importance_weight=importance_weight,
    )

    return loss + sft_loss_ratio * sft_loss, stats


@dataclasses.dataclass
class SPPOActorInterface(model_api.ModelInterface):
    n_minibatches: int = 4

    # Use dict here to allow argument passing through commandline.
    generation_config: Dict = dataclasses.field(default_factory=dict)

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

    enable_save: bool = True

    value_norm: bool = False
    value_norm_type: str = dataclasses.field(
        metadata={"choices": ["exp", "ma"]}, default="exp"
    )
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5

    sft_loss_ratio: float = 1.0

    def __post_init__(self):
        if self.adaptive_kl_ctl:
            assert self.adaptive_kl_target is not None
            assert self.adaptive_kl_horizon is not None
            self.kl_adapter = ppo_functional.AdaptiveKLController(
                self.kl_ctl, self.adaptive_kl_target, self.adaptive_kl_horizon
            )
        else:
            self.kl_adapter = ppo_functional.FixedKLController(self.kl_ctl)
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

        self.gconfig = model_api.GenerationHyperparameters(**self.generation_config)

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
        self,
        model: model_api.Model,
        input_: SequenceSample,
        n_mbs=None,
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
            gconfig=self.gconfig,
            num_micro_batches=n_mbs,
        )
        if res is None:
            return None

        gen_tokens, logprobs, logits_mask = res

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
            prompt_lengths=torch.tensor(flat2d(input_.seqlens["packed_prompts"])).to(
                model.device
            ),
            gen_tokens=gen_tokens,
            logprobs=logprobs,
            logits_mask=logits_mask,
            gen_lengths=gen_lengths,
        )
        gen_strings = model.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False, errors='replace')
        logger.info(f"#########################################GENERATED STRING###############################################\n{gen_strings[0]}\n")
        seqlens = [[s] for s in seq_lengths.cpu().numpy().tolist()]
        data = dict(
            seq_no_eos_mask=seq_no_eos_mask,
            packed_input_ids=packed_input_ids,
            packed_logprobs=packed_logprobs,
            prompt_mask=prompt_mask,
        )
        if not self.gconfig.force_no_logits_mask:
            data["packed_logits_mask"] = packed_logits_mask.bool()
        res = SequenceSample.from_default(
            ids=input_.ids,
            seqlens=seqlens,
            data=data,
        )
        return res

    @torch.no_grad()
    def inference(
        self,
        model: model_api.Model,
        input_: SequenceSample,
        n_mbs=None,
    ) -> SequenceSample:
        module = model.module
        module.eval()

        # This post_hook will gather log probabilities in mini-batches,
        # reducing peak memory usage.
        def calc_logprobs(logits, input_):
            logits /= self.gconfig.temperature
            if (
                "packed_logits_mask" in input_.data
                and input_.data["packed_logits_mask"] is not None
            ):
                apply_logits_mask(logits, input_.data["packed_logits_mask"])

            input_lens = torch.tensor(input_.seqlens["packed_input_ids"]).view(-1)
            cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()

            logprobs = gather_packed_shifted_log_probs(
                logits, cu_seqlens, input_.data["packed_input_ids"]
            )
            return logprobs

        logprobs = module.forward(
            input_=input_,
            num_micro_batches=n_mbs,
            post_hook=calc_logprobs,
        )

        if logprobs is None:
            return None

        res = SequenceSample.from_default(
            ids=input_.ids,
            seqlens=input_.seqlens["packed_input_ids"],
            data=dict(packed_ref_logprobs=logprobs),
        )
        return res

    def train_step(
        self, model: model_api.Model, input_: SequenceSample, n_mbs=None
    ) -> Dict:
        module = model.module
        # We call module.eval() because dropout causes the computation of incorrect of log probs.
        module.eval()

        packed_ref_input_ids = input_.data["packed_ref_input_ids"]
        ref_lengths = input_.seqlens["packed_ref_input_ids"]
        ref_prompt_mask = input_.data["ref_prompt_mask"]

        old_logp: torch.FloatTensor = input_.data["packed_logprobs"].float()
        ref_logp: torch.FloatTensor = input_.data["packed_ref_logprobs"].float()
        prompt_mask = input_.data["prompt_mask"]
        input_lens = torch.tensor(
            flat2d(input_.seqlens["packed_input_ids"]), device=model.device
        )
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
        kl_rewards, rewards = ppo_functional.get_packed_rewards(
            kl_ctl=self.kl_adapter.value,
            clip_reward_value=self.max_reward_clip,
            log_probs=old_logp,
            ref_log_probs=ref_logp,
            reward_score=reward_score,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )
        advantages, returns = ppo_functional.get_packed_advantages_and_returns(
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
                sppo_loss_mask=loss_mask,
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
        if n_mbs is None:
            n_mbs = 1

        ref_prompt_mask_data = SequenceSample.from_default(
            ids=input_.ids,
            data=dict(
                ref_prompt_mask=ref_prompt_mask,
            ),
            seqlens=[len[0] if isinstance(len, list) else len for len in ref_lengths],
        )
        
        input_.update_(ref_prompt_mask_data)

        input_.data["packed_input_ids"] = torch.cat(
            (input_.data["packed_input_ids"], packed_ref_input_ids)
        )
        input_.seqlens["packed_input_ids"] = input_.seqlens["packed_input_ids"] + [[len] if isinstance(len, int) else len for len in ref_lengths]

        datas = [input_]

        ### Logging code starts. ###
        _n_seqs = torch.tensor(
            [reward_score.shape[0]], dtype=torch.float32, device=model.device
        )
        _n_tokens = loss_mask.count_nonzero()
        _n_sft_tokens = ref_prompt_mask.count_nonzero()
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

        # Run mini-batched SPPO training!
        train_stats = collections.defaultdict(lambda: 0)
        for data in datas:
            stats = module.train_batch(
                input_=data,
                version_steps=model.version.global_step,
                num_micro_batches=n_mbs,
                loss_fn=functools.partial(
                    _sppo_actor_loss_from_model_outputs,
                    kl_adapter=self.kl_adapter,
                    eps_clip=self.eps_clip,
                    early_stop_imp_ratio=self.early_stop_imp_ratio,
                    early_stop_kl=self.early_stop_kl,
                    sft_loss_ratio=self.sft_loss_ratio,
                ),
            )

            if stats:
                for k, v in stats.items():
                    train_stats[k] += v
        cur_epoch = model.version.epoch
        model.inc_version()

        # FIXME: It only logs the MoE aux loss of the final SPPO mini-batch.
        global_stats.update(
            constants.log_global_stats_tracker(
                return_dict=True, clear_stats_after_logging=True
            )
        )
        if train_stats:
            train_stats = dict(
                sppo_approx_kl=float(train_stats["sppo_approx_kl"] / _n_tokens),
                actor_loss=float(train_stats["actor_loss"] / _n_tokens),
                sft_loss=float(train_stats["sft_loss"] / _n_sft_tokens),
                sft_ppl=float(train_stats["sft_ppl"] / _n_seqs),
                actor_clip_ratio=float(train_stats["actor_clip_ratio"] / _n_tokens),
                importance_weight=float(train_stats["importance_weight"] / _n_tokens),
            )
            train_stats = dict(**train_stats, **global_stats)

        return dict(train_stats)


model_api.register_interface("sppo_actor", SPPOActorInterface)
