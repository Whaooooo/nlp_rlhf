import itertools
import json
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.utils.data

from realhf.api.core import data_api


class CostModelingPairedDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: int,
        max_pairs_per_prompt: int = 2,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
    ):
        """Dataset used for reward modeling. Each sample consists of a prompt,
        several positive answers, and several negative answers.

        Args:
            util (api.data.DatasetUtility): Dataset utility.
            max_length (int): The maximum sequence length. Sequences will be right-padded to this length
            dataset_path (Optional[str], optional): Path to the dataset json/jsonl file.
                The json/jsonl file should be a list of dictionary. Each element in the list should have
                the key "prompt", "pos_answers", and "neg_answers". Each "pos_answer" must correspond to
                one and only one "neg_answer" (i.e., they are one-to-one pairs). Defaults to None.
            dataset_builder (Optional[Callable[[], List[Dict]]], optional): Alternative to dataset_path.
                A callable that returns a list of dictionary. Defaults to None.
        """
        self._util = util
        seed = util.seed
        tokenizer = util.tokenizer

        self.max_pairs_per_prompt = max_pairs_per_prompt

        self.rng = np.random.RandomState(seed=seed)
        data = data_api.load_shuffle_split_dataset(util, dataset_path, dataset_builder)

        prompts = [x["prompt"] for x in data]
        self.ids = [x["id"] for x in data]

        pos_answers = [
            [x["prompt"] + c + tokenizer.eos_token for c in x["pos_answers"]]
            for x in data
        ]
        neg_answers = [
            [x["prompt"] + c + tokenizer.eos_token for c in x["neg_answers"]]
            for x in data
        ]
        pos_correct = [x["pos_correct"] for x in data]
        neg_correct = [x["neg_correct"] for x in data]

        # Check that pos_answers and neg_answers are one-to-one pairs and non-empty
        for a, b in zip(pos_answers, neg_answers):
            if len(a) != len(b):
                raise RuntimeError(
                    "pos_answers and neg_answers must be one-to-one pairs."
                )
            if len(a) == 0:
                raise RuntimeError("pos_answers and neg_answers must be non-empty.")

        
        if len(pos_correct) != len(neg_correct):
            raise RuntimeError(
                "pos_correct and neg_correct must be aligned with pos_answers and neg_answers."
            )
        if len(pos_correct) == 0 or len(neg_correct) == 0:
            raise RuntimeError("pos_correct and neg_correct must be non-empty.")

        group_sizes = [len(x) for x in pos_answers]

        self.prompt_tokens = tokenizer(
            prompts,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_length=True,
        )
        _pos_answer_tokens = tokenizer(
            list(itertools.chain.from_iterable(pos_answers)),
            max_length=max_length,
            padding=False,
            truncation=True,
            return_length=True,
        )
        _neg_answer_tokens = tokenizer(
            list(itertools.chain.from_iterable(neg_answers)),
            max_length=max_length,
            padding=False,
            truncation=True,
            return_length=True,
        )

        pos_answer_tokens = []
        neg_answer_tokens = []
        grouped_pos_correct = []
        grouped_neg_correct = []
        offset = 0
        for g in group_sizes:
            pos_answer_tokens.append(
                {k: v[offset : offset + g] for k, v in _pos_answer_tokens.items()}
            )
            neg_answer_tokens.append(
                {k: v[offset : offset + g] for k, v in _neg_answer_tokens.items()}
            )
            grouped_pos_correct.append(pos_correct[offset:offset + g])
            grouped_neg_correct.append(neg_correct[offset:offset + g])
            offset += g

        self.pos_answer_tokens: List[Dict[str, List[int]]] = pos_answer_tokens
        self.neg_answer_tokens: List[Dict[str, List[int]]] = neg_answer_tokens
        self.grouped_pos_correct: List[List[int]] = grouped_pos_correct
        self.grouped_neg_correct: List[List[int]] = grouped_neg_correct

        assert (
            len(self.prompt_tokens["input_ids"])
            == len(self.pos_answer_tokens)
            == len(self.neg_answer_tokens)
            == len(self.grouped_pos_correct)
            == len(self.grouped_neg_correct)
        )

    @property
    def util(self):
        return self._util

    def __len__(self):
        return len(self.pos_answer_tokens)

    def __getitem__(self, idx):
        # Sample a piece of data composed of a single prompt
        # and a group of pos-neg answer pairs.

        prompt_len = self.prompt_tokens["length"][idx]
        n_pairs_this_prompt = len(self.pos_answer_tokens[idx]["input_ids"])
        # Randomly select a maximum number of `self.max_pairs_per_prompt` pairs for this prompt
        group_size = min(self.max_pairs_per_prompt, n_pairs_this_prompt)
        pair_indices = self.rng.choice(n_pairs_this_prompt, group_size, replace=False)

        packed_input_ids = []
        input_lens = []
        is_correct = []
        
        for i in pair_indices:
            # Append pos_answer tokens and correctness
            packed_input_ids += self.pos_answer_tokens[idx]["input_ids"][i]
            is_correct.append(self.grouped_pos_correct[idx][i])
            input_lens.append(len(self.pos_answer_tokens[idx]["input_ids"][i]))
            
            # Append neg_answer tokens and correctness
            packed_input_ids += self.neg_answer_tokens[idx]["input_ids"][i]
            is_correct.append(self.grouped_neg_correct[idx][i])
            input_lens.append(len(self.neg_answer_tokens[idx]["input_ids"][i]))

        data = dict(
            packed_input_ids=torch.tensor(packed_input_ids, dtype=torch.long),
            prompt_lens=torch.tensor([prompt_len], dtype=torch.int32),
            is_correct=torch.tensor(is_correct, dtype=torch.int32)
        )

        x = data_api.SequenceSample(
            keys=["packed_input_ids", "prompt_lens", "is_correct"],
            data=data,
            dtypes=dict(
                packed_input_ids=torch.long,
                prompt_lens=torch.int32,
                is_correct=torch.int32,
            ),
            trailing_shapes=dict(
                packed_input_ids=(),
                prompt_lens=(),
                is_correct=(),
            ),
            ids=[self.ids[idx]],
            seqlens=dict(
                packed_input_ids=[input_lens],
                prompt_lens=[[1]],
                is_correct=[[2]],
            ),
        )

        return x


data_api.register_dataset("cost_pair", CostModelingPairedDataset)
