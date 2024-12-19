import json
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.utils.data

from realhf.api.core import data_api
from realhf.base import logging

logger = logging.getLogger("Multi Round Dataset")


class MultiRoundDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: int,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
        pad_to_max_length: bool = False,
    ):
        """A dataset with prompts and corresponding answers. Usually used for SFT.

        Args:
            util (api.data.DatasetUtility): .
            max_length (Optional[int], optional): The maximum length of each sequence in the batch.
            dataset_path (Optional[str], optional): Path to the dataset json/jsonl file.
            dataset_builder (Optional[Callable[[], List[Dict]]], optional): Alternative to dataset_path.
            pad_to_max_length (bool): Whether to pad sequences to the maximum length.
        """
        self._util = util
        tokenizer = self.util.tokenizer

        data = data_api.load_shuffle_split_dataset(util, dataset_path, dataset_builder)

        def combine_to_sentence(user: list, assis: list):
            assert len(user) == len(assis), (len(user), len(assis))
            l = len(user)
            sentence = [tokenizer.bos_token_id]
            mask = [1]
            for i in range(l):
                sentence += user[i] + assis[i]
                mask += [1 for i in range(len(user[i]))] + [0 for i in range(len(assis[i]))]
            return sentence, mask

        self.ids = [x["id"] for x in data]

        prompt_token = []
        prompt_mask = []

        for item in data:
            user_tokens = tokenizer(
                item['user_messages'],
                padding=False,
                truncation=False,  # Disable truncation here
                return_attention_mask=False,
                add_special_tokens=False,
                max_length=1e20,
            )["input_ids"]
            assis_tokens = tokenizer(
                item['assistant_messages'],
                padding=False,
                truncation=False,  # Disable truncation here
                return_attention_mask=False,
                add_special_tokens=False,
                max_length=1e20,
            )["input_ids"]
            token, mask = combine_to_sentence(user_tokens, assis_tokens)
            
            # Manually truncate the combined token list and mask if they exceed max_length
            if len(token) > max_length:
                token = token[:max_length]
                mask = mask[:max_length]
            
            prompt_token.append(token)
            prompt_mask.append(mask)

        self.tokens = prompt_token
        self.prompt_masks = prompt_mask
        
        legnths = [len(x) for x in self.tokens]
        logger.info(
            f"Loaded MultiRound Dataset with INFO: "
            f"#seqs={len(self)}, "
            f"truncation length={max_length}, "
            f"max length = {np.max(legnths)}"
            f'average_length={np.mean(legnths)}, '
        )

    @property
    def util(self):
        return self._util

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        d = {
            "packed_input_ids": torch.tensor(
                self.tokens[idx], dtype=torch.long
            ),
            "prompt_mask": torch.tensor(self.prompt_masks[idx], dtype=torch.bool),
        }
        assert len(d["packed_input_ids"]) == len(d["prompt_mask"])
        seqlen = [len(d["packed_input_ids"])]
        x = data_api.SequenceSample.from_default(
            ids=[self.ids[idx]],
            seqlens=seqlen,
            data=d,
        )
        return x


data_api.register_dataset("multi_round", MultiRoundDataset)
