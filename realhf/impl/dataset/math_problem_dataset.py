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

        seqs = [x["prompt"] + x["answer"] + util.tokenizer.eos_token for x in data]
        prompts_str = [x["prompt"] for x in data]
        target_str = [x["target"] for x in data]

        self.ids = [x["id"] for x in data]
        util.tokenizer.padding_side = "left"

        tokens = util.tokenizer(
            seqs,
            truncation=True,
            max_length=max_length,
            return_length=True,
            return_attention_mask=False,
            padding=False,
        )
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


        self.token_lengths = tokens["length"]
        self.tokens = tokens["input_ids"]
        self.prompt_lengths = prompt_encodings["length"]
        self.prompts = prompt_encodings["input_ids"]
        self.target_lengths = target_encodings["length"]
        self.targets = target_encodings["input_ids"]

        prompt_masks = []
        for i in range(len(self)):
            prompt_len = self.prompt_lengths[i]
            seqlen = self.token_lengths[i]
            # seq = self.tokens["input_ids"][i]
            # prompt = prompt_tokens["input_ids"][i]
            # assert seq[:prompt_len] == prompt, (seq, prompt, prompt_len, seqlen)
            assert seqlen >= prompt_len, (seqlen, prompt_len)
            prompt_mask = [1] * prompt_len + [0] * (seqlen - prompt_len)
            prompt_masks.append(prompt_mask)

        self.prompt_masks = prompt_masks


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
            keys=["packed_prompts", "packed_targets", "packed_ref_input_ids", "ref_prompt_mask"],
            data=dict(
                packed_prompts=torch.tensor(self.prompts[idx], dtype=torch.long), 
                packed_targets=torch.tensor(self.targets[idx], dtype=torch.long),
                packed_ref_input_ids=torch.tensor(self.tokens[idx], dtype=torch.long),
                ref_prompt_mask=torch.tensor(self.prompt_masks[idx], dtype=torch.bool),
            ),
            dtypes=dict(
                packed_prompts=torch.long,
                packed_targets=torch.long,
                packed_ref_input_ids=torch.long,
                ref_prompt_mask=torch.bool,
            ),
            trailing_shapes=dict(
                packed_prompts=(),
                packed_targets=(),
                packed_ref_input_ids=(),
                ref_prompt_mask=(),
            ),
            ids=[self.ids[idx]],
            seqlens=dict(
                packed_prompts=[[self.prompt_lengths[idx]]],
                packed_targets=[[self.target_lengths[idx]]],
                packed_ref_input_ids=[[self.token_lengths[idx]]],
                ref_prompt_mask=[[self.token_lengths[idx]]],
            ),
            metadata=dict(random_id=[uuid.uuid4()]),
        )
    

data_api.register_dataset("math_problem", MathProblemDataset)
