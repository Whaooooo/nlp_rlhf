import uuid
from typing import Callable, Dict, List, Optional

import torch.utils.data

from realhf.api.core import data_api
from realhf.base import logging

logger = logging.getLogger("Math Problem Dataset")


class MathProblemDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: Optional[int] = None,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
    ):
        """A dataset with math problems and their answers. Usually used for CPPO.

        Args:
            util (api.data.DatasetUtility): .
            max_length (Optional[int], optional): The maximum length of each sequence in the batch.
            dataset_path (Optional[str], optional): Path to the dataset json/jsonl file.
                The json/jsonl file should be a list of dictionary. Each element in the list should have
                a key "prompt". Defaults to None.
            dataset_builder (Optional[Callable[[], List[Dict]]], optional): Alternative to dataset_path.
                A callable that returns a list of dictionary. Defaults to None.
        """
        self._util = util
        self.max_length = max_length

        data = data_api.load_shuffle_split_dataset(util, dataset_path, dataset_builder)

        prompts_str = [x["prompt"] for x in data]
        target_str = [x["target"] for x in data]
        self.ids = [x["id"] for x in data]
        util.tokenizer.padding_side = "left"
        prompt_encodings = util.tokenizer(
            prompts_str,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_length=True,
            return_attention_mask=False,
        )
        target_encodings = util.tokenizer(
            target_str,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_length=True,
            return_attention_mask=False,
        )

        self.prompt_lengths = prompt_encodings["length"]
        self.prompts = prompt_encodings["input_ids"]
        self.target_lengths = target_encodings["length"]
        self.targets = target_encodings["input_ids"]
        assert all(len(x) == l for x, l in zip(self.prompts, self.prompt_lengths))
        assert all(len(x) == l for x, l in zip(self.targets, self.target_lengths))

        logger.info(f"Number of prompts in the dataset: {len(self.prompts)}")

    @property
    def util(self):
        return self._util

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return data_api.SequenceSample(
            keys=["packed_prompts", "packed_targets"],
            data=dict(
                packed_prompts=torch.tensor(self.prompts[idx], dtype=torch.long), 
                packed_targets=torch.tensor(self.targets[idx], dtype=torch.long)
            ),
            dtypes=dict(
                packed_prompts=torch.long,
                packed_targets=torch.long,
            ),
            trailing_shapes=dict(
                packed_prompts=(),
                packed_targets=(),
            ),
            ids=[self.ids[idx]],
            seqlens=dict(
                packed_prompts=[torch.tensor([self.prompt_lengths[idx]], dtype=torch.int32)], 
                packed_targets=[torch.tensor([self.target_lengths[idx]], dtype=torch.int32)]
            ),
            metadata=dict(random_id=[uuid.uuid4()]),
        )
    

data_api.register_dataset("math_problem", MathProblemDataset)
