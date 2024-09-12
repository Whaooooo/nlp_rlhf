import dataclasses
import itertools
import tqdm
from typing import Dict, Optional, Tuple


import torch
import torch.distributed as dist

import realhf.api.core.model_api as model_api
import realhf.base.constants as constants
import realhf.base.logging as logging
import realhf.impl.model.utils.cppo_functional as cppo_functional
from realhf.api.core.data_api import SequenceSample
from realhf.base.datapack import flat2d
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.nn.real_llm_generate import concat_prompt_to_generation_output
from realhf.impl.model.utils.functional import (
    apply_logits_mask,
    gather_packed_shifted_log_probs,
    masked_normalization,
)

logger = logging.getLogger("PackedCOSTInterface")


def flatten_list(l):
    return list(itertools.chain(*l))

def _cost_loss_from_model_outputs(
    scores: torch.FloatTensor,
    input_: SequenceSample,
):
    group_sizes = [len(x) // 2 for x in input_.seqlens["packed_input_ids"]]
    assert all([x >= 1 for x in group_sizes])
    group_factor = torch.tensor(
        flatten_list([[1 / g for _ in range(g)] for g in group_sizes]),
        device=scores.device,
    )

    input_lens = torch.tensor(flat2d(input_.seqlens["packed_input_ids"]))

    assert scores.shape[0] == input_lens.sum(), (scores.shape, input_lens.sum())
    scores = scores[input_lens.cumsum(0) - 1].view(-1, 2).float()
    avg_score = torch.mean(scores)
    is_correct = 2 * input_.data["is_correct"].clone().detach().to(device=scores.device).view(-1, 2).float() - 1

    loss = -(
        (torch.nn.functional.logsigmoid(scores[:, 0] - scores[:, 1])
        + torch.nn.functional.logsigmoid(is_correct[:, 0] * (scores[:, 0] - avg_score))
        + torch.nn.functional.logsigmoid(is_correct[:, 1] * (scores[:, 1] - avg_score))
        ) * group_factor
    ).sum()

    flat_scores = scores.view(-1)
    flat_is_correct = is_correct.view(-1)

    reward_accuracy = (
        ((flat_is_correct == 1) & (flat_scores > avg_score))
        | ((flat_is_correct == -1) & (flat_scores < avg_score))
    ).float().sum().detach()

    correct_score = flat_scores[flat_is_correct == 1]
    incorrect_score = flat_scores[flat_is_correct == -1]

    correct_score_sum = correct_score.sum().detach()
    incorrect_score_sum = incorrect_score.sum().detach()

    correct_score_count = correct_score.numel()
    incorrect_score_count = incorrect_score.numel()

    avg_correct_score = correct_score_sum / correct_score_count if correct_score_count > 0 else 0.0
    avg_incorrect_score = incorrect_score_sum / incorrect_score_count if incorrect_score_count > 0 else 0.0

    max_correct_score = correct_score.max().detach() if correct_score_count > 0 else torch.tensor(float('-inf'), device=scores.device)
    min_incorrect_score = incorrect_score.min().detach() if incorrect_score_count > 0 else torch.tensor(float('inf'), device=scores.device)

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
        reward_accuracy,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )
    dist.all_reduce(
        total_predictions,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )

    dist.all_reduce(
        correct_score_sum,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )
    dist.all_reduce(
        incorrect_score_sum,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )
    dist.all_reduce(
        loss.detach(),
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )
    dist.all_reduce(
        max_correct_score,
        op=dist.ReduceOp.MAX,
        group=constants.data_parallel_group(),
    )
    dist.all_reduce(
        min_incorrect_score,
        op=dist.ReduceOp.MIN,
        group=constants.data_parallel_group(),
    )

    return loss, dict(
        loss=loss.detach(),
        correct_predictions=correct_predictions,
        total_predictions=total_predictions,
        reward_accuracy=reward_accuracy,
        avg_correct_score=avg_correct_score,
        avg_incorrect_score=avg_incorrect_score,
        max_correct_score=max_correct_score,
        min_incorrect_score=min_incorrect_score,
        correct_score_sum=correct_score_sum,
        incorrect_score_sum=incorrect_score_sum,
        correct_score_count=correct_score_count,
        incorrect_score_count=incorrect_score_count,
    )

@dataclasses.dataclass
class CostInterface(model_api.ModelInterface):
    enable_save: bool = True

    output_scaling: float = 1.0
    output_bias: float = 0.0

    # training log
    train_total_predictions: int = 0
    train_total_correct_predictions: int = 0

    @torch.no_grad()
    def inference(
        self, model: model_api.Model, data: SequenceSample, n_mbs=None
    ) -> SequenceSample:

        module = model.module

        module.eval()

        r = module.forward(input_=data, num_micro_batches=n_mbs)
        if r is None:
            return
        scores = r.float()

        input_lens = torch.tensor(flat2d(data.seqlens["packed_input_ids"]))
        scores = scores.view(-1)[input_lens.cumsum(0) - 1].float()  # [bs]
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
                    [1 for _ in range(len(x))] for x in data.seqlens["packed_input_ids"]
                ]
            ),
            data=dict(rewards=scores),
        )
        return res
    
    
    def train_step(
        self, model: model_api.Model, data: SequenceSample, n_mbs=None
    ) -> SequenceSample:
        module = model.module
        module.train()

        stats = module.train_batch(
            input_=data,
            loss_fn=_cost_loss_from_model_outputs,
            version_steps=model.version.global_step,
            num_micro_batches=n_mbs,
        )

        res = {}
        global_stats = constants.log_global_stats_tracker(
            return_dict=True, clear_stats_after_logging=True
        )
        if stats:
            if constants.pipe_parallel_world_size() > 1:
                stats["max_correct_score"] /= constants.pipe_parallel_world_size() * 2
                stats["min_incorrect_score"] /= constants.pipe_parallel_world_size() * 2
            self.train_total_predictions += int(stats["total_predictions"])
            self.train_total_correct_predictions += int(stats["correct_predictions"])
            res = dict(
                loss=float(stats["loss"] / stats["total_predictions"]),
                epoch_acc=self.train_total_correct_predictions
                / self.train_total_predictions,
                batch_acc=float(
                    stats["correct_predictions"] / stats["total_predictions"]
                ),
                reward_acc=float(stats["reward_accuracy"] / (2 * stats["total_predictions"])),
                avg_correct_score=float(stats["avg_correct_score"]),
                avg_incorrect_score=float(stats["avg_incorrect_score"]),
                max_correct_score=float(stats["max_correct_score"]),
                min_incorrect_score=float(stats["min_incorrect_score"]),
                correct_score_sum=float(stats["correct_score_sum"]),
                incorrect_score_sum=float(stats["incorrect_score_sum"]),
                total_predictions=int(stats["total_predictions"]),
                correct_predictions=int(stats["correct_predictions"]),
                **global_stats,
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
        reward_accuracy = 0  # Initialize reward accuracy counter
        losses = 0
        correct_score_sum = incorrect_score_sum = 0
        max_correct_score = -float("inf")
        min_incorrect_score = float("inf")
        avg_correct_score = 0.0
        avg_incorrect_score = 0.0
        total_correct_count = 0  # Count of correct examples for averaging
        total_incorrect_count = 0  # Count of incorrect examples for averaging

        for step, data in enumerate(tqdm.tqdm(eval_dataloader)):
            data: SequenceSample
            stats = model.eval_batch(
                input_=data.cuda(),
                loss_fn=_cost_loss_from_model_outputs,
            )

            if stats:
                losses += stats["loss"].item()
                correct_predictions += stats["correct_predictions"].item()
                reward_accuracy += stats["reward_accuracy"].item()  # Accumulate reward accuracy
                total_predictions += stats["total_predictions"].item()

                correct_score_sum += stats["correct_score_sum"].item()
                incorrect_score_sum += stats["incorrect_score_sum"].item()

                max_correct_score = max(max_correct_score, stats["max_correct_score"].item())
                min_incorrect_score = min(min_incorrect_score, stats["min_incorrect_score"].item())

                total_correct_count += stats["correct_score_count"]
                total_incorrect_count += stats["incorrect_score_count"]

        global_stats = constants.log_global_stats_tracker(
            return_dict=True, clear_stats_after_logging=True
        )

        if total_predictions > 0:
            avg_correct_score = correct_score_sum / total_correct_count if total_correct_count > 0 else 0.0
            avg_incorrect_score = incorrect_score_sum / total_incorrect_count if total_incorrect_count > 0 else 0.0

            return dict(
                loss=float(losses / total_predictions),
                acc=correct_predictions / total_predictions,
                reward_acc=reward_accuracy / (2 * total_predictions),  # Compute reward accuracy
                avg_correct_score=avg_correct_score,
                avg_incorrect_score=avg_incorrect_score,
                correct_score_sum=correct_score_sum,
                incorrect_score_sum=incorrect_score_sum,
                max_correct_score=float(max_correct_score),
                min_incorrect_score=float(min_incorrect_score),
                correct_predictions=int(correct_predictions),
                total_predictions=int(total_predictions),
                **global_stats,
            )
        return dict()

model_api.register_interface("paired_cost", CostInterface)
