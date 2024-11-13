import dataclasses
from typing import Optional

import colorama
import torch
import transformers

from realhf.api.core import model_api
from realhf.api.core.config import (
    ModelAbstraction,
    ModelBackendAbstraction,
    ModelInterfaceAbstraction,
)
from realhf.api.core.data_api import SequenceSample
from realhf.api.core.system_api import ExperimentConfig
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.apps.quickstart import main
from realhf.base import logging
from realhf.base.datapack import flat2d
from realhf.experiments.common.ppo_exp import PPOConfig

logger = logging.getLogger("Sentiment PPO example")


class SentimentScoringInterface(model_api.ModelInterface):

    def __post_init__(self):
        # Paths to the models and tokenizers
        armo_model_path = "/home/zzo/Quickstart/asset/model/models--RLHFlow--ArmoRM-Llama3-8B-v0.1/snapshots/eb2676d20da2f2d41082289d23c59b9f7427f955"
        generator_tokenizer_path = "/home/zzo/Quickstart/asset/model/models--mistralai--Mistral-7B-v0.3/snapshots/e676bf786d9d83284bd571f785f068e5d1f0c9f9"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the ARMO model
        self.score_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            armo_model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.score_model.eval()

        # Load the ARMO tokenizer
        self.score_tokenizer = transformers.AutoTokenizer.from_pretrained(
            armo_model_path,
            use_fast=True,
        )

        # Load the generator's tokenizer
        self.generator_tokenizer = transformers.AutoTokenizer.from_pretrained(
            generator_tokenizer_path,
            use_fast=True,
        )

        # Load any additional configurations or attributes if necessary
        self.attributes = [
            'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
            'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
            'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
            'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
            'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm', 'code-complexity',
            'code-style', 'code-explanation', 'code-instruction-following', 'code-readability'
        ]

    @torch.no_grad()
    def inference(
        self,
        model: model_api.Model,
        input_: SequenceSample,
        n_mbs: Optional[int] = None,
    ) -> SequenceSample:
        device = model.device
        packed_input_ids: torch.Tensor = input_.data["packed_input_ids"]
        seqlens_cpu = torch.tensor(flat2d(input_.seqlens["packed_input_ids"]))
        max_seqlen = int(max(seqlens_cpu))
        bs = input_.bs

        # Build attention mask.
        _ind = torch.arange(max_seqlen, dtype=torch.long, device=device)
        attention_mask = _ind.unsqueeze(0) < seqlens_cpu.to(device).unsqueeze(1)

        # Pad input_ids.
        input_ids = torch.full(
            (bs, max_seqlen),
            fill_value=model.tokenizer.pad_token_id,
            device=device,
            dtype=torch.long,
        )
        for i in range(bs):
            seq_len = seqlens_cpu[i]
            input_ids[i, :seq_len] = packed_input_ids[i * max_seqlen: i * max_seqlen + seq_len]

        # Decode input_ids using the generator's tokenizer.
        texts = self.generator_tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        # Prepare messages for each sample in the batch.
        messages_list = []
        for text in texts:
            # Split the text to extract user and assistant messages.
            # Assuming the delimiters are 'BEGINNING OF CONVERSATION: USER:' and 'ASSISTANT:'
            # and that the assistant's reply is at the end.
            user_delimiter = 'BEGINNING OF CONVERSATION: USER:'
            assistant_delimiter = 'ASSISTANT:'

            # Find the positions of the delimiters.
            try:
                user_start = text.index(user_delimiter) + len(user_delimiter)
                assistant_start = text.index(assistant_delimiter, user_start) + len(assistant_delimiter)
            except ValueError:
                raise ValueError("Delimiters not found in the input text.")

            # Extract user and assistant messages.
            user_content = text[user_start:assistant_start - len(assistant_delimiter)].strip()
            assistant_content = text[assistant_start:].strip()

            # Construct the messages.
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
            messages_list.append(messages)

        # Tokenize using the ARMO tokenizer with the chat template.
        encoding = self.score_tokenizer.apply_chat_template(
            messages_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # Perform inference using the ARMO model.
        output = self.score_model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )

        # Extract the preference score.
        preference_score = output.score.cpu().float().squeeze()
        assert preference_score.shape == (bs,), preference_score.shape

        # Prepare the result.
        res = SequenceSample.from_default(
            ids=input_.ids,
            seqlens=[1 for _ in range(bs)],
            data=dict(rewards=preference_score),
        )
        return res




model_api.register_interface("sentiment_scoring", SentimentScoringInterface)


class MyPPOConfig(PPOConfig):

    def initial_setup(self) -> ExperimentConfig:
        if (
            self.rew_inf.parallel.model_parallel_size > 1
            or self.rew_inf.parallel.pipeline_parallel_size > 1
        ):
            raise ValueError(
                "For this example, the reward model does not support model parallel or pipeline parallel."
            )

        cfg = super().initial_setup()

        # Replace the backend and model configurations for the reward model.
        for mw in cfg.model_worker:
            for s in mw.shards:
                if s.id.model_name.role == "reward":
                    s.model = ModelAbstraction(
                        "tokenizer",
                        args=dict(
                            tokenizer_path=self.rew.path,
                        ),
                    )
                    s.backend = ModelBackendAbstraction("null")

        # Change the model function call implementation.
        idx = 0
        for rpc in cfg.model_rpcs:
            if rpc.model_name.role == "reward":
                break
            idx += 1
        inf_reward_rpc = cfg.model_rpcs[idx]
        inf_reward_rpc.interface_impl = ModelInterfaceAbstraction("sentiment_scoring")
        inf_reward_rpc.post_hooks = []

        return cfg


register_quickstart_exp("my-ppo", MyPPOConfig)

if __name__ == "__main__":
    main()
