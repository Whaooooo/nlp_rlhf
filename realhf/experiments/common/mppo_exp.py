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
from realhf.api.quickstart.dataset import MathProblemDatasetConfig
from realhf.api.quickstart.device_mesh import DeviceMesh, MFCConfig, RPCAllocation
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import ModelTrainEvalConfig, ParallelismConfig
from realhf.experiments.common.common import CommonExperimentConfig

logger = logging.getLogger("MPPO exp", "colored")


@dataclasses.dataclass
class MPPOHyperparameters:

    gen: GenerationHyperparameters = dataclasses.field(
        default_factory=GenerationHyperparameters
    )
    mppo_n_minibatches: int = 4
    kl_ctl: float = 0.1

    discount: float = 1.0
    gae_lambda: float = 1.0
    eps_clip: float = 0.2
    value_eps_clip: float = 0.2

    max_reward_clip: float = 20.0
    reward_output_scaling: float = 1.0
    reward_output_bias: float = 0.0

    max_cost1_clip: float = 20.0
    cost1_output_scaling: float = 1.0
    cost1_output_bias: float = 0.0

    max_cost2_clip: float = 20.0
    cost2_output_scaling: float = 1.0
    cost2_output_bias: float = 0.0

    dual1_ratio: float = 1.0
    dual1_beta: float = 0.9
    dual1_epsilon: float = 1e-5
    dual1_lr: float = 0.1
    dual1_target: Optional[float] = None

    dual2_ratio: float = 1.0
    dual2_beta: float = 0.9
    dual2_epsilon: float = 1e-5
    dual2_lr: float = 0.1
    dual2_target: Optional[float] = None

    early_stop_imp_ratio: float = 5.0
    use_adaptive_kl_ctl: bool = False
    adv_norm: bool = True
    value_norm: bool = True
    value_norm_type: str = dataclasses.field(
        metadata={"choices": ["exp", "ma"]}, default="exp" 
    )
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5


@dataclasses.dataclass
class MPPOConfig(CommonExperimentConfig):

    is_sft_lora: bool = False
    sft_lora_path: Optional[str] = None
    is_rew_lora: bool = False
    rew_lora_path: Optional[str] = None
    rew_head_path: Optional[str] = None

    actor: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    rew_critic: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    cost1_critic: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    cost2_critic: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    ref: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    rew: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    cost1: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    cost2: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)

    # for manual allocation only
    actor_train: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    rew_critic_train: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    cost1_critic_train: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    cost2_critic_train: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    actor_gen: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    rew_critic_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    cost1_critic_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    cost2_critic_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    rew_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    cost1_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    cost2_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    ref_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)

    dataset: MathProblemDatasetConfig = dataclasses.field(
        default_factory=MathProblemDatasetConfig
    )

    mppo: MPPOHyperparameters = dataclasses.field(default_factory=MPPOHyperparameters)

    def __post_init__(self):
        if self.is_sft_lora or self.sft_lora_path is not None:
            raise NotImplementedError("SFT LoRA is not sumpported yet.")
        if self.is_rew_lora or self.rew_lora_path is not None:
            raise NotImplementedError("Rew LoRA is not sumpported yet.")

        self.mppo_kwargs = dict(
            n_minibatches=self.mppo.mppo_n_minibatches,
            kl_ctl=self.mppo.kl_ctl,
            discount=self.mppo.discount,
            gae_lambda=self.mppo.gae_lambda,
            eps_clip=self.mppo.eps_clip,
            value_eps_clip=self.mppo.value_eps_clip,
            adaptive_kl_ctl=self.mppo.use_adaptive_kl_ctl,
            value_norm=self.mppo.value_norm,
            value_norm_type=self.mppo.value_norm_type,
            value_norm_beta=self.mppo.value_norm_beta,
            value_norm_eps=self.mppo.value_norm_eps,
        )

        if self.mppo.gen.use_cuda_graph and (
            self.actor_train.parallel != self.actor_gen.parallel
        ):
            raise ValueError(
                "CUDA graph cannot be used with parameter reallocation "
                "because CUDA graph requires pinned parameter memory. "
                "Either set use_cuda_graph=False or set identical parallel "
                "strategies for actor_train and actor_gen."
            )

    @property
    def models(self) -> Dict[str, ModelTrainEvalConfig]:
        # role to config
        return {
            "actor": self.actor,
            "rew_critic": self.rew_critic,
            "cost1_critic": self.cost1_critic,
            "cost2_critic": self.cost2_critic,
            "ref": self.ref,
            "reward": self.rew,
            "cost1": self.cost1,
            "cost2": self.cost2,
        }

    @property
    def rpcs(self):
        # interfaces
        actor_interface = ModelInterfaceAbstraction(
            "mppo_actor",
            args={
                **copy.deepcopy(self.mppo_kwargs),
                "generation_config": OmegaConf.to_container(self.mppo.gen, resolve=True),
                "early_stop_imp_ratio": self.mppo.early_stop_imp_ratio,
                "adv_norm": self.mppo.adv_norm,
                "dual1_ratio": self.mppo.dual1_ratio,
                "dual1_beta": self.mppo.dual1_beta,
                "dual1_epsilon": self.mppo.dual1_epsilon,
                "dual1_lr": self.mppo.dual1_lr,
                "dual1_target": self.mppo.dual1_target,
                "dual2_ratio": self.mppo.dual2_ratio,
                "dual2_beta": self.mppo.dual2_beta,
                "dual2_epsilon": self.mppo.dual2_epsilon,
                "dual2_lr": self.mppo.dual2_lr,
                "dual2_target": self.mppo.dual2_target,
            },
        )
        ref_interface = copy.deepcopy(actor_interface)
        ref_interface.args["enable_save"] = False

        rew_critic_interface = ModelInterfaceAbstraction(
            "rcppo_rew_critic",
            args={
                **copy.deepcopy(self.mppo_kwargs),
                "max_reward_clip": self.mppo.max_reward_clip,
            }
        )
        rew_critic_interface.args.pop("eps_clip")
        rew_critic_interface.args["enable_save"] = False

        cost1_critic_interface = ModelInterfaceAbstraction(
            "rcppo_rew_critic",
            args={
                **copy.deepcopy(self.mppo_kwargs),
                "max_reward_clip": self.mppo.max_cost1_clip,
            }
        )
        cost1_critic_interface.args.pop("eps_clip")
        cost1_critic_interface.args["enable_save"] = False

        cost2_critic_interface = ModelInterfaceAbstraction(
            "rcppo_rew_critic",
            args={
                **copy.deepcopy(self.mppo_kwargs),
                "max_reward_clip": self.mppo.max_cost2_clip,
            }
        )
        cost2_critic_interface.args.pop("eps_clip")
        cost2_critic_interface.args["enable_save"] = False

        rw_interface = ModelInterfaceAbstraction(
            "mppo_rw",
            args=dict(
                enable_save=False,
                output_scaling=self.mppo.reward_output_scaling,
                output_bias=self.mppo.reward_output_bias,
            ),
        )

        cost1_interface = ModelInterfaceAbstraction(
            "mppo_rw",
            args=dict(
                enable_save=False,
                output_scaling=self.mppo.cost1_output_scaling,
                output_bias=self.mppo.cost1_output_bias,
            ),
        )

        cost2_interface = ModelInterfaceAbstraction(
            "mppo_gt_rw",
            args=dict(
                enable_save=False,
                output_scaling=self.mppo.cost2_output_scaling,
                output_bias=self.mppo.cost2_output_bias,
            ),
        )

        rollout = MFCDef(
            name="actor_gen",
            model_name="actor",
            n_mbs=self.actor_gen.n_mbs,
            interface_type=ModelInterfaceType.GENERATE,
            model_type=self.actor.type,
            model_path=self.actor.path,
            interface_impl=actor_interface,
            input_keys=["packed_prompts", "packed_targets"],
            output_keys=[
                "seq_no_eos_mask",
                "packed_input_ids",
                "packed_logprobs",
                "prompt_mask",
                "packed_logits_mask",
            ],
            balanced_dp=True,
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_reward = MFCDef(
            name="rew_inf",
            model_name="reward",
            n_mbs=self.rew_inf.n_mbs,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=rw_interface,
            model_type=self.rew.type,
            model_path=self.rew.path,
            input_keys=["packed_input_ids", "packed_targets"],
            output_keys=["rewards"],
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_cost1 = MFCDef(
            name="cost1_inf",
            model_name="cost1",
            n_mbs=self.cost1_inf.n_mbs,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=cost1_interface,
            model_type=self.cost1.type,
            model_path=self.cost1.path,
            input_keys=["packed_input_ids", "packed_targets"],
            output_keys=["costs1"],
            output_key_remap={"rewards": "costs1"},
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_cost2 = MFCDef(
            name="cost2_inf",
            model_name="cost2",
            n_mbs=self.cost2_inf.n_mbs,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=cost2_interface,
            model_type=self.cost2.type,
            model_path=self.cost2.path,
            input_keys=["packed_input_ids", "packed_targets"],
            output_keys=["costs2"],
            output_key_remap={"rewards": "costs2"},
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_ref_inputs = ["packed_input_ids"]
        if not self.mppo.gen.force_no_logits_mask:
            inf_ref_inputs.append(
                "packed_logits_mask",
            )
        inf_ref_logits = MFCDef(
            name="ref_inf",
            model_name="ref",
            n_mbs=self.ref_inf.n_mbs,
            interface_type=ModelInterfaceType.INFERENCE,
            model_type=self.ref.type,
            model_path=self.ref.path,
            interface_impl=ref_interface,
            input_keys=inf_ref_inputs,
            output_keys=["packed_ref_logprobs"],
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_rew_values = MFCDef(
            name="rew_critic_inf",
            model_name="rew_critic",
            n_mbs=self.rew_critic_inf.n_mbs,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=rew_critic_interface,
            model_type=self.rew_critic.type,
            model_path=self.rew_critic.path,
            input_keys=["packed_input_ids", "seq_no_eos_mask"],
            output_keys=["rew_values"],
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_cost1_values = MFCDef(
            name="cost1_critic_inf",
            model_name="cost1_critic",
            n_mbs=self.cost1_critic_inf.n_mbs,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=cost1_critic_interface,
            model_type=self.cost1_critic.type,
            model_path=self.cost1_critic.path,
            input_keys=["packed_input_ids", "seq_no_eos_mask"],
            output_keys=["cost1_values"],
            output_key_remap={"rewards": "cost1_values"},
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_cost2_values = MFCDef(
            name="cost2_critic_inf",
            model_name="cost2_critic",
            n_mbs=self.cost2_critic_inf.n_mbs,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=cost2_critic_interface,
            model_type=self.cost2_critic.type,
            model_path=self.cost2_critic.path,
            input_keys=["packed_input_ids", "seq_no_eos_mask"],
            output_keys=["cost2_values"],
            output_key_remap={"rewards": "cost2_values"},
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        train_actor_inputs = [
            "packed_input_ids",
            "packed_logprobs",
            "packed_ref_logprobs",
            "rewards",
            "costs1",
            "costs2",
            "rew_values",
            "cost1_values",
            "cost2_values",
            "prompt_mask",
            "seq_no_eos_mask",
            "packed_logits_mask",
            "signal_rew",
            "signal_cost1",
            "signal_cost2",
        ]
        if self.mppo.gen.force_no_logits_mask:
            train_actor_inputs.remove("packed_logits_mask")
        train_actor = MFCDef(
            name="actor_train",
            model_name="actor",
            n_mbs=self.actor_train.n_mbs,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            model_type=self.actor.type,
            model_path=self.actor.path,
            interface_impl=actor_interface,
            input_keys=train_actor_inputs,
            log_return_value=True,
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        train_rew_critic = MFCDef(
            name="rew_critic_train",
            model_name="rew_critic",
            n_mbs=self.rew_critic_train.n_mbs,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=rew_critic_interface,
            model_type=self.rew_critic.type,
            model_path=self.rew_critic.path,
            input_keys=[
                "packed_input_ids",
                "packed_logprobs",
                "packed_ref_logprobs",
                "rewards",
                "rew_values",
                "prompt_mask",
                "seq_no_eos_mask",
            ],
            output_keys=["signal_rew"],
            log_return_value=True,
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        train_cost1_critic = MFCDef(
            name="cost1_critic_train",
            model_name="cost1_critic",
            n_mbs=self.cost1_critic_train.n_mbs,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=cost1_critic_interface,
            model_type=self.cost1_critic.type,
            model_path=self.cost1_critic.path,
            input_keys=[
                "packed_input_ids",
                "packed_logprobs",
                "packed_ref_logprobs",
                "costs1",
                "cost1_values",
                "prompt_mask",
                "seq_no_eos_mask",
            ],
            input_key_remap={"cost1_values": "rew_values", "costs1": "rewards"},
            output_keys=["signal_cost1"],
            log_return_value=True,
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        train_cost2_critic = MFCDef(
            name="cost2_critic_train",
            model_name="cost2_critic",
            n_mbs=self.cost2_critic_train.n_mbs,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=cost2_critic_interface,
            model_type=self.cost2_critic.type,
            model_path=self.cost2_critic.path,
            input_keys=[
                "packed_input_ids",
                "packed_logprobs",
                "packed_ref_logprobs",
                "costs2",
                "cost2_values",
                "prompt_mask",
                "seq_no_eos_mask",
            ],
            input_key_remap={"cost2_values": "rew_values", "costs2": "rewards"},
            output_keys=["signal_cost2"],
            log_return_value=True,
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        return {
            "actor_gen": rollout,
            "actor_train": train_actor,
            "rew_critic_inf": inf_rew_values,
            "rew_critic_train": train_rew_critic,
            "cost1_critic_inf": inf_cost1_values,
            "cost2_critic_inf": inf_cost2_values,
            "cost1_critic_train": train_cost1_critic,
            "cost2_critic_train": train_cost2_critic,
            "ref_inf": inf_ref_logits,
            "rew_inf": inf_reward,
            "cost1_inf": inf_cost1,
            "cost2_inf": inf_cost2,
        }

    @property
    def allocations(self):
        return {
            "actor_gen": self.actor_gen,
            "actor_train": self.actor_train,
            "rew_critic_inf": self.rew_critic_inf,
            "rew_critic_train": self.rew_critic_train,
            "cost1_critic_inf": self.cost1_critic_inf,
            "cost2_critic_inf": self.cost2_critic_inf,
            "cost1_critic_train": self.cost1_critic_train,
            "cost2_critic_train": self.cost2_critic_train,
            "ref_inf": self.ref_inf,
            "rew_inf": self.rew_inf,
            "cost1_inf": self.cost1_inf,
            "cost2_inf": self.cost2_inf,
        }

    @property
    def datasets(self):
        return [
            DatasetAbstraction(
                "math_problem",
                args=dict(
                    dataset_path=self.dataset.path,
                    max_length=self.dataset.max_prompt_len,
                ),
            )
        ]

    @property
    def tokenizer_name_or_path(self) -> str:
        return self.actor.path

    @property
    def search_kwargs(self):
        return {
            "num_gen_tokens": self.mppo.gen.max_new_tokens,
            "n_mppo_minibatches": self.mppo.mppo_n_minibatches,
            "seq_len": self.dataset.max_prompt_len,
        }

    @property
    def max_prompt_len(self):
        return self.dataset.max_prompt_len

    def _heuristic_rpc_allocation(self):
        """Heuristic RPC allocation based on the user's specifications."""
        assert self.n_nodes == 1
        assert self.n_gpus_per_node == 8

        allocations = []

        # rew_critic_inf and rew_critic_train on GPUs 0 and 1
        rew_critic_mapping = np.array([[1, 1, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
        rew_critic_device_mesh = DeviceMesh(
            n_nodes=1,
            n_gpus_per_node=8,
            mapping=rew_critic_mapping,
            global_mesh_name=self.nodelist,
        )
        rew_critic_parallel = ParallelismConfig(
            data_parallel_size=1,
            pipeline_parallel_size=1,
            model_parallel_size=2,  # Number of GPUs occupied
        )
        rew_critic_inf = RPCAllocation(
            rpc=self.rpcs["rew_critic_inf"],
            device_mesh=rew_critic_device_mesh,
            parallel=rew_critic_parallel,
        )
        rew_critic_train = RPCAllocation(
            rpc=self.rpcs["rew_critic_train"],
            device_mesh=rew_critic_device_mesh,
            parallel=rew_critic_parallel,
        )
        allocations.extend([rew_critic_inf, rew_critic_train])

        # cost1_critic_inf and cost1_critic_train on GPUs 2 and 3
        cost1_critic_mapping = np.array([[0, 0, 1, 1, 0, 0, 0, 0]], dtype=np.int32)
        cost1_critic_device_mesh = DeviceMesh(
            n_nodes=1,
            n_gpus_per_node=8,
            mapping=cost1_critic_mapping,
            global_mesh_name=self.nodelist,
        )
        cost1_critic_parallel = ParallelismConfig(
            data_parallel_size=1,
            pipeline_parallel_size=1,
            model_parallel_size=2,
        )
        cost1_critic_inf = RPCAllocation(
            rpc=self.rpcs["cost1_critic_inf"],
            device_mesh=cost1_critic_device_mesh,
            parallel=cost1_critic_parallel,
        )
        cost1_critic_train = RPCAllocation(
            rpc=self.rpcs["cost1_critic_train"],
            device_mesh=cost1_critic_device_mesh,
            parallel=cost1_critic_parallel,
        )
        allocations.extend([cost1_critic_inf, cost1_critic_train])

        # cost2_critic_inf and cost2_critic_train on GPUs 4 and 5
        cost2_critic_mapping = np.array([[0, 0, 0, 0, 1, 1, 0, 0]], dtype=np.int32)
        cost2_critic_device_mesh = DeviceMesh(
            n_nodes=1,
            n_gpus_per_node=8,
            mapping=cost2_critic_mapping,
            global_mesh_name=self.nodelist,
        )
        cost2_critic_parallel = ParallelismConfig(
            data_parallel_size=1,
            pipeline_parallel_size=1,
            model_parallel_size=2,
        )
        cost2_critic_inf = RPCAllocation(
            rpc=self.rpcs["cost2_critic_inf"],
            device_mesh=cost2_critic_device_mesh,
            parallel=cost2_critic_parallel,
        )
        cost2_critic_train = RPCAllocation(
            rpc=self.rpcs["cost2_critic_train"],
            device_mesh=cost2_critic_device_mesh,
            parallel=cost2_critic_parallel,
        )
        allocations.extend([cost2_critic_inf, cost2_critic_train])

        # rew_inf, cost1_inf, cost2_inf on GPUs 6 and 7
        inf_mapping = np.array([[0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.int32)
        inf_device_mesh = DeviceMesh(
            n_nodes=1,
            n_gpus_per_node=8,
            mapping=inf_mapping,
            global_mesh_name=self.nodelist,
        )
        inf_parallel = ParallelismConfig(
            data_parallel_size=1,
            pipeline_parallel_size=1,
            model_parallel_size=2,
        )
        rew_inf = RPCAllocation(
            rpc=self.rpcs["rew_inf"],
            device_mesh=inf_device_mesh,
            parallel=inf_parallel,
        )
        cost1_inf = RPCAllocation(
            rpc=self.rpcs["cost1_inf"],
            device_mesh=inf_device_mesh,
            parallel=inf_parallel,
        )
        cost2_inf = RPCAllocation(
            rpc=self.rpcs["cost2_inf"],
            device_mesh=inf_device_mesh,
            parallel=inf_parallel,
        )
        allocations.extend([rew_inf, cost1_inf, cost2_inf])

        # Allocate actor_gen, actor_train, and ref_inf across all 8 GPUs
        full_mapping = np.ones((1, 8), dtype=np.int32)
        full_device_mesh = DeviceMesh(
            n_nodes=1,
            n_gpus_per_node=8,
            mapping=full_mapping,
            global_mesh_name=self.nodelist,
        )
        actor_parallel = ParallelismConfig(
            data_parallel_size=1,
            pipeline_parallel_size=1,
            model_parallel_size=8,
        )
        actor_gen = RPCAllocation(
            rpc=self.rpcs["actor_gen"],
            device_mesh=full_device_mesh,
            parallel=actor_parallel,
        )
        actor_train = RPCAllocation(
            rpc=self.rpcs["actor_train"],
            device_mesh=full_device_mesh,
            parallel=actor_parallel,
        )
        ref_inf = RPCAllocation(
            rpc=self.rpcs["ref_inf"],
            device_mesh=full_device_mesh,
            parallel=actor_parallel,
        )
        allocations.extend([actor_gen, actor_train, ref_inf])

        return allocations


register_quickstart_exp("mppo", MPPOConfig)
