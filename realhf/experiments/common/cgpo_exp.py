import copy
import dataclasses
import math
from typing import *

import numpy as np
from omegaconf import OmegaConf

import realhf.base.logging as logging
from realhf.api.core.config import (
    DatasetAbstraction,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
)
from realhf.api.core.dfg import MFCDef
from realhf.api.core.model_api import GenerationHyperparameters
from realhf.api.quickstart.dataset import PromptOnlyDatasetConfig
from realhf.api.quickstart.device_mesh import DeviceMesh, MFCConfig, RPCAllocation
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import ModelTrainEvalConfig, ParallelismConfig
from realhf.experiments.common.common import CommonExperimentConfig

logger = logging.getLogger("CGPO exp", "colored")


@dataclasses.dataclass
class CGPOHyperparameters:
    """Configuration for CGPO hyperparameters."""

    gen: GenerationHyperparameters = dataclasses.field(
        default_factory=GenerationHyperparameters
    )
    cgpo_n_minibatches: int = 4
    kl_ctl: float = 0.1
    discount: float = 1.0
    gae_lambda: float = 1.0
    eps_clip: float = 0.2
    value_eps_clip: float = 0.2
    max_reward_clip: float = 20.0
    reward_output_scaling: float = 1.0
    reward_output_bias: float = 0.0
    early_stop_imp_ratio: float = 5.0
    use_adaptive_kl_ctl: bool = False
    adv_norm: bool = True
    value_norm: bool = True
    value_norm_type: str = dataclasses.field(
        metadata={"choices": ["exp", "ma"]}, default="exp"
    )
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5
    no_eos_penalty: Optional[float] = None


@dataclasses.dataclass
class TaskConfig:
    """Configuration for one task in CGPO experiments."""

    actor: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    critic: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    reward: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    dataset: PromptOnlyDatasetConfig = dataclasses.field(
        default_factory=PromptOnlyDatasetConfig
    )
    cgpo: CGPOHyperparameters = dataclasses.field(default_factory=CGPOHyperparameters)
    cgpo_kwargs: dict = dataclasses.field(default_factory=dict)
    # For manual allocation
    actor_train: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    critic_train: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    actor_gen: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    critic_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    rew_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)


@dataclasses.dataclass
class CGPOConfig(CommonExperimentConfig):
    """Configuration for CGPO experiments.

    This class is a subclass of :class:`CommonExperimentConfig`,
    so all CLI options from the base class are available.

    We support up to 6 tasks. Each task has its own actor, critic, reward, dataset, and hyperparameters.
    All tasks share the same reference model (`ref`).

    :param task_number: The number of tasks (up to 6).
    :type task_number: int
    """

    task_number: int = 1  # Default to 1 task, maximum 6

    is_sft_lora: bool = False
    sft_lora_path: Optional[str] = None
    is_rew_lora: bool = False
    rew_lora_path: Optional[str] = None
    rew_head_path: Optional[str] = None

    # Up to 6 tasks
    task0: TaskConfig = dataclasses.field(default_factory=TaskConfig)
    task1: TaskConfig = dataclasses.field(default_factory=TaskConfig)
    task2: TaskConfig = dataclasses.field(default_factory=TaskConfig)
    task3: TaskConfig = dataclasses.field(default_factory=TaskConfig)
    task4: TaskConfig = dataclasses.field(default_factory=TaskConfig)
    task5: TaskConfig = dataclasses.field(default_factory=TaskConfig)

    ref: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    ref_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)

    def __post_init__(self):
        if self.is_sft_lora or self.sft_lora_path is not None:
            raise NotImplementedError("SFT LoRA is not supported yet.")
        if self.is_rew_lora or self.rew_lora_path is not None:
            raise NotImplementedError("Rew LoRA is not supported yet.")
        if self.task_number > 6 or self.task_number < 1:
            raise ValueError("task_number must be between 1 and 6")
        tasks = [self.task0, self.task1, self.task2, self.task3, self.task4, self.task5]
        for i in range(self.task_number):
            tasks[i].cgpo_kwargs = dict(
                n_minibatches=tasks[i].cgpo.cgpo_n_minibatches,
                kl_ctl=tasks[i].cgpo.kl_ctl,
                discount=tasks[i].cgpo.discount,
                gae_lambda=tasks[i].cgpo.gae_lambda,
                eps_clip=tasks[i].cgpo.eps_clip,
                value_eps_clip=tasks[i].cgpo.value_eps_clip,
                max_reward_clip=tasks[i].cgpo.max_reward_clip,
                adaptive_kl_ctl=tasks[i].cgpo.use_adaptive_kl_ctl,
                value_norm=tasks[i].cgpo.value_norm,
                value_norm_type=tasks[i].cgpo.value_norm_type,
                value_norm_beta=tasks[i].cgpo.value_norm_beta,
                value_norm_eps=tasks[i].cgpo.value_norm_eps,
                temperature=tasks[i].cgpo.gen.temperature,
                no_eos_penalty=tasks[i].cgpo.no_eos_penalty
            )

    @property
    def models(self) -> Dict[str, ModelTrainEvalConfig]:
        """Collect models from all tasks and the shared ref model."""
        models_dict = {"ref": self.ref}
        tasks = [self.task0, self.task1, self.task2, self.task3, self.task4, self.task5]
        for i in range(self.task_number):
            task = tasks[i]
            models_dict[f"actor{i}"] = task.actor
            models_dict[f"critic_{i}"] = task.critic  # Note the underscore
            models_dict[f"reward_{i}"] = task.reward  # Note the underscore
        return models_dict

    @property
    def rpcs(self):
        rpcs_dict = {}
        tasks = [self.task0, self.task1, self.task2, self.task3, self.task4, self.task5]
        for i in range(self.task_number):
            task = tasks[i]
            # Actor
            actor_interface = ModelInterfaceAbstraction(
                "cgpo_actor", # Use cpgo_actor instead of ppo_actor, with different INFERENCE and TRAIN_STEP to implement multiple optimizers on actor model
                args={
                    **copy.deepcopy(task.cgpo_kwargs),
                    "generation_config": OmegaConf.to_container(
                        task.cgpo.gen, resolve=True
                    ),
                    "early_stop_imp_ratio": task.cgpo.early_stop_imp_ratio,
                    "adv_norm": task.cgpo.adv_norm,
                },
            )
            if i > 0:
                actor_interface.args["enable_save"] = False
            ref_interface = ModelInterfaceAbstraction(
                "ppo_actor",
                args={
                    **copy.deepcopy(task.cgpo_kwargs),
                    "generation_config": OmegaConf.to_container(
                        task.cgpo.gen, resolve=True
                    ),
                    "early_stop_imp_ratio": task.cgpo.early_stop_imp_ratio,
                    "adv_norm": task.cgpo.adv_norm,
                },
            )
            ref_interface.args["enable_save"] = False

            critic_interface = ModelInterfaceAbstraction(
                "ppo_critic",
                args=copy.deepcopy(task.cgpo_kwargs),
            )
            critic_interface.args.pop("eps_clip")
            critic_interface.args["enable_save"] = False

            rw_interface = ModelInterfaceAbstraction(
                "paired_rw",
                args=dict(
                    enable_save=False,
                    output_scaling=task.cgpo.reward_output_scaling,
                    output_bias=task.cgpo.reward_output_bias,
                ),
            )
            # Rollout
            rollout = MFCDef(
                name=f"actor_gen_{i}",
                model_name=f"actor{i}",
                n_mbs=task.actor_gen.n_mbs,
                interface_type=ModelInterfaceType.GENERATE,
                model_type=task.actor.type,
                model_path=task.actor.path,
                interface_impl=actor_interface,
                input_keys=[f"packed_prompts_{i}"],
                input_key_remap={f"packed_prompts_{i}": "packed_prompts"},
                output_keys=[
                    f"seq_no_eos_mask_{i}",
                    f"packed_input_ids_{i}",
                    f"packed_logprobs_{i}",
                    f"prompt_mask_{i}",
                    f"packed_logits_mask_{i}",
                ],
                output_key_remap={
                    "seq_no_eos_mask": f"seq_no_eos_mask_{i}",
                    "packed_input_ids": f"packed_input_ids_{i}",
                    "packed_logprobs": f"packed_logprobs_{i}",
                    "prompt_mask": f"prompt_mask_{i}",
                    "packed_logits_mask": f"packed_logits_mask_{i}",
                },
                balanced_dp=True,
                n_seqs=task.dataset.train_bs_n_seqs,
            )
            rpcs_dict[f"actor_gen_{i}"] = rollout

            inf_reward = MFCDef(
                name=f"rew_inf_{i}",
                model_name=f"reward_{i}",
                n_mbs=task.rew_inf.n_mbs,
                interface_type=ModelInterfaceType.INFERENCE,
                interface_impl=rw_interface,
                input_keys=[f"packed_input_ids_{i}"],
                input_key_remap={f"packed_input_ids_{i}": "packed_input_ids"},
                output_keys=[f"rewards_{i}"],
                output_key_remap={"rewards": f"rewards_{i}"},
                n_seqs=task.dataset.train_bs_n_seqs,
            )
            rpcs_dict[f"rew_inf_{i}"] = inf_reward

            # InfValues
            inf_values = MFCDef(
                name=f"critic_inf_{i}",
                model_name=f"critic_{i}",
                n_mbs=task.critic_inf.n_mbs,
                interface_type=ModelInterfaceType.INFERENCE,
                interface_impl=critic_interface,
                input_keys=[f"packed_input_ids_{i}", f"seq_no_eos_mask_{i}"],
                input_key_remap={
                    f"packed_input_ids_{i}": "packed_input_ids",
                    f"seq_no_eos_mask_{i}": "seq_no_eos_mask",
                },
                output_keys=[f"values_{i}"],
                output_key_remap={"values": f"values_{i}"},
                n_seqs=task.dataset.train_bs_n_seqs,
            )
            rpcs_dict[f"critic_inf_{i}"] = inf_values

            # Actor Inference
            backward_inputs = [
                f"packed_input_ids_{i}",
                f"packed_logprobs_{i}",
                f"packed_ref_logprobs_{i}",
                f"rewards_{i}",
                f"values_{i}",
                f"prompt_mask_{i}",
                f"seq_no_eos_mask_{i}",
                f"packed_logits_mask_{i}",
            ]
            backward_input_remap = {
                f"packed_input_ids_{i}": "packed_input_ids",
                f"packed_logprobs_{i}": "packed_logprobs",
                f"packed_ref_logprobs_{i}": "packed_ref_logprobs",
                f"rewards_{i}": "rewards",
                f"values_{i}": "values",
                f"prompt_mask_{i}": "prompt_mask",
                f"seq_no_eos_mask_{i}": "seq_no_eos_mask",
                f"packed_logits_mask_{i}": "packed_logits_mask",
            }
            if task.cgpo.gen.force_no_logits_mask:
                backward_inputs.remove(f"packed_logits_mask_{i}")
                backward_input_remap.pop(f"packed_logits_mask_{i}", None)

            actor_backward = MFCDef(
                name=f"actor_backward_{i}",
                model_name=f"actor{i}",
                n_mbs=task.actor_train.n_mbs,
                interface_type=ModelInterfaceType.TRAIN_STEP,
                model_type=task.actor.type,
                model_path=task.actor.path,
                interface_impl=actor_interface,
                input_keys=backward_inputs,
                input_key_remap=backward_input_remap,
                output_keys=[f"backward_signal_{i}"],
                output_key_remap={"backward_signal": f"backward_signal_{i}"},
                log_return_value=False,
                n_seqs=task.dataset.train_bs_n_seqs,
            )
            rpcs_dict[f"actor_backward_{i}"] = actor_backward

            # Actor Train Step
            train_actor_inputs = [
                f"packed_input_ids_{i}",
                f"packed_logprobs_{i}",
                f"packed_ref_logprobs_{i}",
                f"rewards_{i}",
                f"values_{i}",
                f"prompt_mask_{i}",
                f"seq_no_eos_mask_{i}",
                f"packed_logits_mask_{i}",
                *[f"backward_signal_{j}" for j in range(self.task_number)],
            ]
            train_input_remap = {
                f"packed_input_ids_{i}": "packed_input_ids",
                f"packed_logprobs_{i}": "packed_logprobs",
                f"packed_ref_logprobs_{i}": "packed_ref_logprobs",
                f"rewards_{i}": "rewards",
                f"values_{i}": "values",
                f"prompt_mask_{i}": "prompt_mask",
                f"seq_no_eos_mask_{i}": "seq_no_eos_mask",
                f"packed_logits_mask_{i}": "packed_logits_mask",
            }
            if task.cgpo.gen.force_no_logits_mask:
                train_actor_inputs.remove(f"packed_logits_mask_{i}")
                train_input_remap.pop(f"packed_logits_mask_{i}", None)

            actor_train = MFCDef(
                name=f"actor_train_{i}",
                model_name=f"actor{i}",
                n_mbs=task.actor_train.n_mbs,
                interface_type=ModelInterfaceType.INFERENCE,
                model_type=task.actor.type,
                model_path=task.actor.path,
                interface_impl=actor_interface,
                input_keys=train_actor_inputs,
                input_key_remap=train_input_remap,
                log_return_value=True,
                n_seqs=task.dataset.train_bs_n_seqs,
            )
            rpcs_dict[f"actor_train_{i}"] = actor_train

            # TrainCritic
            train_critic_inputs = [
                f"packed_input_ids_{i}",
                f"packed_logprobs_{i}",
                f"packed_ref_logprobs_{i}",
                f"rewards_{i}",
                f"values_{i}",
                f"prompt_mask_{i}",
                f"seq_no_eos_mask_{i}",
            ]
            critic_input_remap = {
                f"packed_input_ids_{i}": "packed_input_ids",
                f"packed_logprobs_{i}": "packed_logprobs",
                f"packed_ref_logprobs_{i}": "packed_ref_logprobs",
                f"rewards_{i}": "rewards",
                f"values_{i}": "values",
                f"prompt_mask_{i}": "prompt_mask",
                f"seq_no_eos_mask_{i}": "seq_no_eos_mask",
            }
            train_critic = MFCDef(
                name=f"critic_train_{i}",
                model_name=f"critic_{i}",
                n_mbs=task.critic_train.n_mbs,
                interface_type=ModelInterfaceType.TRAIN_STEP,
                interface_impl=critic_interface,
                model_type=task.critic.type,
                model_path=task.critic.path,
                input_keys=train_critic_inputs,
                input_key_remap=critic_input_remap,
                log_return_value=True,
                n_seqs=task.dataset.train_bs_n_seqs,
            )
            rpcs_dict[f"critic_train_{i}"] = train_critic

            # Ref Inference for each task
            inf_ref_inputs = [f"packed_input_ids_{i}"]
            inf_ref_input_remap = {f"packed_input_ids_{i}": "packed_input_ids"}
            if not task.cgpo.gen.force_no_logits_mask:
                inf_ref_inputs.append(f"packed_logits_mask_{i}")
                inf_ref_input_remap[f"packed_logits_mask_{i}"] = "packed_logits_mask"

            inf_ref_logits = MFCDef(
                name=f"ref_inf_{i}",
                model_name="ref",
                n_mbs=self.ref_inf.n_mbs,
                interface_type=ModelInterfaceType.INFERENCE,
                model_type=self.ref.type,
                model_path=self.ref.path,
                interface_impl=ref_interface,
                input_keys=inf_ref_inputs,
                input_key_remap=inf_ref_input_remap,
                output_keys=[f"packed_ref_logprobs_{i}"],
                output_key_remap={"packed_ref_logprobs": f"packed_ref_logprobs_{i}"},
                n_seqs=task.dataset.train_bs_n_seqs,
            )
            rpcs_dict[f"ref_inf_{i}"] = inf_ref_logits

        return rpcs_dict

    @property
    def allocations(self):
        raise NotImplementedError

    @property
    def datasets(self):
        tasks = [self.task0, self.task1, self.task2, self.task3, self.task4, self.task5]
        """Collect datasets from all tasks."""
        return [DatasetAbstraction(
            "math_problem",
            args=dict(
                dataset_path=[task.dataset.path for task in tasks[: self.task_number]],
                max_length=tasks[0].dataset.max_prompt_len,
            ),
        )]

    @property
    def tokenizer_name_or_path(self) -> str:
        return self.ref.path  # Assuming all tasks share the same tokenizer

    @property
    def search_kwargs(self):
        """Provide search parameters for experiments."""
        tasks = [self.task0, self.task1, self.task2, self.task3, self.task4, self.task5]
        return {
            "num_gen_tokens": [
                task.cgpo.gen.max_new_tokens for task in tasks[: self.task_number]
            ],
            "n_ppo_minibatches": [
                task.cgpo.cgpo_n_minibatches for task in tasks[: self.task_number]
            ],
            "seq_len": [
                task.dataset.max_prompt_len for task in tasks[: self.task_number]
            ],
        }

    @property
    def max_prompt_len(self):
        tasks = [self.task0, self.task1, self.task2, self.task3, self.task4, self.task5]
        return max(task.dataset.max_prompt_len for task in tasks[: self.task_number])

    def _heuristic_rpc_allocation(self):
        assert self.n_gpus_per_node==8
        """Define allocations for RPCs."""
        allocations = []
        tasks = [self.task0, self.task1, self.task2, self.task3, self.task4, self.task5]
        for i in range(self.task_number):
            task = tasks[i]
            # Allocation for actor_gen_{i}
            actor_gen = RPCAllocation(
                rpc=self.rpcs[f"actor_gen_{i}"],
                device_mesh=DeviceMesh(
                    n_nodes=self.n_nodes,
                    n_gpus_per_node=self.n_gpus_per_node,
                    mapping=np.ones((self.n_nodes, self.n_gpus_per_node), dtype=np.int32),
                    global_mesh_name=self.nodelist,
                ),
                parallel=ParallelismConfig(
                    data_parallel_size=1,
                    pipeline_parallel_size=1,
                    model_parallel_size=8,
                ),
            )
            allocations.append(actor_gen)

            # Allocation for actor_backward_{i}
            actor_backward = RPCAllocation(
                rpc=self.rpcs[f"actor_backward_{i}"],
                device_mesh=DeviceMesh(
                    n_nodes=self.n_nodes,
                    n_gpus_per_node=self.n_gpus_per_node,
                    mapping=np.ones(
                        (self.n_nodes, self.n_gpus_per_node), dtype=np.int32
                    ),
                    global_mesh_name=self.nodelist,
                ),
                parallel=ParallelismConfig(
                    data_parallel_size=1,
                    pipeline_parallel_size=1,
                    model_parallel_size=8,
                ),
            )
            allocations.append(actor_backward)

            # Allocation for actor_train_{i}
            actor_train = RPCAllocation(
                rpc=self.rpcs[f"actor_train_{i}"],
                device_mesh=DeviceMesh(
                    n_nodes=self.n_nodes,
                    n_gpus_per_node=self.n_gpus_per_node,
                    mapping=np.ones(
                        (self.n_nodes, self.n_gpus_per_node), dtype=np.int32
                    ),
                    global_mesh_name=self.nodelist,
                ),
                parallel=ParallelismConfig(
                    data_parallel_size=1,
                    pipeline_parallel_size=1,
                    model_parallel_size=8,
                ),
            )
            allocations.append(actor_train)

            # Allocation for critic_inf_{i}
            critic_inf = RPCAllocation(
                rpc=self.rpcs[f"critic_inf_{i}"],
                device_mesh=DeviceMesh(
                    n_nodes=self.n_nodes,
                    n_gpus_per_node=self.n_gpus_per_node,
                    mapping=np.ones(
                        (self.n_nodes, self.n_gpus_per_node), dtype=np.int32
                    ),
                    global_mesh_name=self.nodelist,
                ),
                parallel=ParallelismConfig(
                    data_parallel_size=1,
                    pipeline_parallel_size=1,
                    model_parallel_size=8,
                ),
            )
            allocations.append(critic_inf)

            # Allocation for critic_train_{i}
            critic_train = RPCAllocation(
                rpc=self.rpcs[f"critic_train_{i}"],
                device_mesh=DeviceMesh(
                    n_nodes=self.n_nodes,
                    n_gpus_per_node=self.n_gpus_per_node,
                    mapping=np.ones(
                        (self.n_nodes, self.n_gpus_per_node), dtype=np.int32
                    ),
                    global_mesh_name=self.nodelist,
                ),
                parallel=ParallelismConfig(
                    data_parallel_size=1,
                    pipeline_parallel_size=1,
                    model_parallel_size=8,
                ),
            )
            allocations.append(critic_train)

            # Allocation for rew_inf_{i}
            rew_inf = RPCAllocation(
                rpc=self.rpcs[f"rew_inf_{i}"],
                device_mesh=DeviceMesh(
                    n_nodes=self.n_nodes,
                    n_gpus_per_node=self.n_gpus_per_node,
                    mapping=np.ones(
                        (self.n_nodes, self.n_gpus_per_node), dtype=np.int32
                    ),
                    global_mesh_name=self.nodelist,
                ),
                parallel=ParallelismConfig(
                    data_parallel_size=1,
                    pipeline_parallel_size=1,
                    model_parallel_size=8,
                ),
            )
            allocations.append(rew_inf)

            # Allocation for ref_inf_{i}
            ref_inf = RPCAllocation(
                rpc=self.rpcs[f"ref_inf_{i}"],
                device_mesh=DeviceMesh(
                    n_nodes=self.n_nodes,
                    n_gpus_per_node=self.n_gpus_per_node,
                    mapping=np.ones((self.n_nodes, self.n_gpus_per_node), dtype=np.int32),
                    global_mesh_name=self.nodelist,
                ),
                parallel=ParallelismConfig(
                    data_parallel_size=1,
                    pipeline_parallel_size=1,
                    model_parallel_size=8,
                ),
            )
            allocations.append(ref_inf)

        return allocations


register_quickstart_exp("cgpo", CGPOConfig)
