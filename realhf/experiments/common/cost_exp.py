import dataclasses
from typing import List, Optional

from realhf.api.core.config import (
    DataLoaderAbstraction,
    DatasetAbstraction,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
    ModelName,
)
from realhf.api.core.dfg import MFCDef
from realhf.api.quickstart.dataset import PairedComparisonDatasetConfig
from realhf.api.quickstart.device_mesh import MFCConfig
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import ModelTrainEvalConfig
from realhf.experiments.common.common import CommonExperimentConfig


@dataclasses.dataclass
class COSTConfig(CommonExperimentConfig):
    """Pairwise reward modeling experiment configuration.

    It is a subclass of :class:`CommonExperimentConfig`,
    so all CLI options in the base class are available.

    :param is_sft_lora: Whether LoRA was used for SFT.
        If so, the saved SFT model should only contain LoRA parameters.
        Since LoRA is currently not supported for SFT,
        this option is not used for now.
    :type is_sft_lora: bool
    :param sft_lora_path: Path to the LoRA model for SFT.
        Since LoRA is currently not supported for SFT,
        this option is not used for now.
    :param model: Model runtime configuration.
    :type model: ModelTrainEvalConfig
    :param allocation: Device allocation and parallelism configuration.
    :type allocation: MFCConfig
    :param dataset: Dataset configuration.
    :type dataset: PairedComparisonDatasetConfig
    :param n_mbs: Number of microbatches.
    :type n_mbs: int
    """

    is_sft_lora: bool = False
    sft_lora_path: Optional[str] = None
    model: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    allocation: MFCConfig = dataclasses.field(default_factory=MFCConfig)

    dataset: PairedComparisonDatasetConfig = dataclasses.field(
        default_factory=PairedComparisonDatasetConfig
    )

    def __post_init__(self):
        assert (
            not self.is_sft_lora and self.sft_lora_path is None
        ), "LoRA is not supported for now."
        self.model.init_critic_from_actor = True

    @property
    def models(self):
        return {
            "default": self.model,
        }

    @property
    def rpcs(self):
        interface = ModelInterfaceAbstraction("paired_cost")
        rpc = MFCDef(
            name="rwTrain",
            n_mbs=self.allocation.n_mbs,
            model_name=ModelName("default", 0),
            interface_type=ModelInterfaceType.TRAIN_STEP, 
            interface_impl=interface,
            model_type=self.model.type,
            model_path=self.model.path,
            input_keys=["packed_input_ids", "is_correct"],
            log_return_value=True,
            n_seqs=self.dataset.train_bs_n_seqs,
        )
        return {"rwTrain": rpc}

    @property
    def allocations(self):
        return {"rwTrain": self.allocation}

    @property
    def datasets(self):
        return [
            DatasetAbstraction(
                "cost_pair",
                args=dict(
                    max_length=self.dataset.max_seqlen,
                    max_pairs_per_prompt=self.dataset.max_pairs_per_prompt,
                    dataset_path=self.dataset.train_path,
                ),
            )
        ]

    @property
    def eval_datasets(self):
        return [
            DatasetAbstraction(
                "cost_pair",
                args=dict(
                    max_length=self.dataset.max_seqlen,
                    max_pairs_per_prompt=self.dataset.max_pairs_per_prompt,
                    dataset_path=self.dataset.valid_path,
                ),
            )
        ]

    @property
    def eval_dataloader(self):
        return DataLoaderAbstraction(
            "packed_eval", args=dict(batch_size=self.dataset.valid_bs_n_seqs)
        )

    @property
    def tokenizer_name_or_path(self):
        return self.model.path


register_quickstart_exp("cost", COSTConfig)
