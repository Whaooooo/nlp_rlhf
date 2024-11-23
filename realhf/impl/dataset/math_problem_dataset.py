import uuid
import json
from typing import Callable, Dict, List, Optional, Union

import torch.utils.data

from realhf.api.core import data_api
from realhf.api.core.data_api import DatasetUtility, get_shuffle_indices
from realhf.base import logging

logger = logging.getLogger("Math Problem Dataset")


class MathProblemDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: Optional[int] = None,
        dataset_path: Optional[Union[str, List[str]]] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
    ):
        """A dataset with math problems and their answers. Usually used for CPPO.

        Args:
            util (api.data.DatasetUtility): Dataset utility instance.
            max_length (Optional[int], optional): The maximum length of each sequence in the batch.
            dataset_path (Optional[str or List[str]], optional): Path or list of paths to the dataset json/jsonl files.
                Each json/jsonl file should be a list of dictionaries. Each element in the list should have
                a key "prompt". Defaults to None.
            dataset_builder (Optional[Callable[[], List[Dict]]], optional): Alternative to dataset_path.
                A callable that returns a list of dictionaries. Defaults to None.
        """
        self._util = util
        self.max_length = max_length

        data_list = data_api.load_shuffle_split_dataset(util, dataset_path, dataset_builder)

        # If data_list is a single data list, wrap it in a list to make processing uniform
        if isinstance(data_list[0], dict):
            data_list = [data_list]

        self.ids = [x["id"] for x in data_list[0]]  # Use ids from the first data list

        self.token_lengths_list = []
        self.tokens_list = []
        self.prompt_lengths_list = []
        self.prompts_list = []
        self.target_lengths_list = []
        self.targets_list = []
        self.prompt_masks_list = []

        util.tokenizer.padding_side = "left"

        for data in data_list:
            seqs = [x["prompt"] + x["answer"] for x in data]
            prompts_str = [x["prompt"] for x in data]
            target_str = [x["target"] for x in data]

            tokens = util.tokenizer(
                seqs,
                truncation=True,
                max_length=max_length,
                return_length=True,
                return_attention_mask=False,
                padding=False,
                add_special_tokens=False,
            )
            prompt_encodings = util.tokenizer(
                prompts_str,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_length=True,
                return_attention_mask=False,
                add_special_tokens=False,
            )
            target_encodings = util.tokenizer(
                target_str,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_length=True,
                return_attention_mask=False,
                add_special_tokens=False,
            )

            token_lengths = tokens["length"]
            tokens_ = tokens["input_ids"]
            prompt_lengths = prompt_encodings["length"]
            prompts_ = prompt_encodings["input_ids"]
            target_lengths = target_encodings["length"]
            targets_ = target_encodings["input_ids"]

            prompt_masks = []
            for i in range(len(data)):
                prompt_len = prompt_lengths[i]
                seqlen = token_lengths[i]
                assert seqlen >= prompt_len, (seqlen, prompt_len)
                prompt_mask = [1] * prompt_len + [0] * (seqlen - prompt_len)
                prompt_masks.append(prompt_mask)

            self.token_lengths_list.append(token_lengths)
            self.tokens_list.append(tokens_)
            self.prompt_lengths_list.append(prompt_lengths)
            self.prompts_list.append(prompts_)
            self.target_lengths_list.append(target_lengths)
            self.targets_list.append(targets_)
            self.prompt_masks_list.append(prompt_masks)

        logger.info(f"Number of prompts in the dataset: {len(self.ids)}")

    @property
    def util(self):
        return self._util

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        num_datasets = len(self.token_lengths_list)
        keys = []
        data = {}
        dtypes = {}
        trailing_shapes = {}
        seqlens = {}

        for i in range(num_datasets):
            tokens = self.tokens_list[i][idx]
            prompts = self.prompts_list[i][idx]
            targets = self.targets_list[i][idx]
            token_lengths = self.token_lengths_list[i][idx]
            prompt_lengths = self.prompt_lengths_list[i][idx]
            target_lengths = self.target_lengths_list[i][idx]
            prompt_masks = self.prompt_masks_list[i][idx]

            key_suffix = f"_{i}" if num_datasets > 1 else ""  # Add index suffix if multiple datasets

            keys.extend([
                f"packed_prompts{key_suffix}",
                f"packed_targets{key_suffix}",
                f"packed_ref_input_ids{key_suffix}",
                f"ref_prompt_mask{key_suffix}",
            ])
            data.update({
                f"packed_prompts{key_suffix}": torch.tensor(prompts, dtype=torch.long),
                f"packed_targets{key_suffix}": torch.tensor(targets, dtype=torch.long),
                f"packed_ref_input_ids{key_suffix}": torch.tensor(tokens, dtype=torch.long),
                f"ref_prompt_mask{key_suffix}": torch.tensor(prompt_masks, dtype=torch.bool),
            })
            dtypes.update({
                f"packed_prompts{key_suffix}": torch.long,
                f"packed_targets{key_suffix}": torch.long,
                f"packed_ref_input_ids{key_suffix}": torch.long,
                f"ref_prompt_mask{key_suffix}": torch.bool,
            })
            trailing_shapes.update({
                f"packed_prompts{key_suffix}": (),
                f"packed_targets{key_suffix}": (),
                f"packed_ref_input_ids{key_suffix}": (),
                f"ref_prompt_mask{key_suffix}": (),
            })
            seqlens.update({
                f"packed_prompts{key_suffix}": [[prompt_lengths]],
                f"packed_targets{key_suffix}": [[target_lengths]],
                f"packed_ref_input_ids{key_suffix}": [[token_lengths]],
                f"ref_prompt_mask{key_suffix}": [[token_lengths]],
            })

        return data_api.SequenceSample(
            keys=keys,
            data=data,
            dtypes=dtypes,
            trailing_shapes=trailing_shapes,
            ids=[self.ids[idx]],
            seqlens=seqlens,
            metadata=dict(random_id=[uuid.uuid4()]),
        )


data_api.register_dataset("math_problem", MathProblemDataset)