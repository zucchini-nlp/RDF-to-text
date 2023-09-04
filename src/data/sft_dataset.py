import copy
import logging
from dataclasses import dataclass
from typing import Dict, List

import datasets
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

from src.data.data_utils import make_data_module

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

class SFTInstructionDataset(Dataset):
    """
    Dataset for supervised fine-tuning of instruction following models.

    Converts raw dataset containing source/target instructions
    into tokenized input/target pairs with truncation and padding.

    Attributes:
        dataset: The raw dataset containing source/target examples
        tokenizer: Tokenizer to use for encoding text
        max_seq_len: Maximum sequence length for truncation

    """

    def __init__(
        self,
        raw_data: datasets.DatasetDict,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 1024,
    ):
        self.dataset = raw_data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len


    def __len__(self) -> int:
        return len(self.dataset)


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.dataset[idx]

        source_text = example["input"]
        target_text = example["output"]

        # if several reference texts were provided as output
        if isinstance(target_text, list):
            target_text = target_text[0]
        
        text = (
            f"{self.tokenizer.bos_token}{source_text}{target_text}{self.tokenizer.eos_token}"
        )

        tokenized_source = self.tokenizer(
            source_text,
            max_length=self.max_seq_len,
            padding=True,
            add_special_tokens=False,
        )

        tokenized = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding=True,
            add_special_tokens=False,
        )

        # https://github.com/huggingface/transformers/issues/22794#issuecomment-1601482558
        input_ids = tokenized["input_ids"]
        source_ids = tokenized_source['input_ids']

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]

        masked_source = [IGNORE_INDEX for _ in range(len(source_ids))]
        target_ids = copy.deepcopy(input_ids[len(source_ids): ])
        labels = torch.tensor((masked_source + target_ids)[:self.max_seq_len])
        input_ids = torch.tensor(input_ids)

        data_dict = {"input_ids": input_ids, "labels": labels}
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset:
    """
    Collate and pad examples for supervised training.
    """

    tokenizer: PreTrainedTokenizer
    predict_with_generate: bool = False

    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate examples into dictionary for supervised training.

        Args:
            examples: List of examples, each containing 'input_ids' and 'labels'

        Returns:
            Dictionary with padded 'input_ids', 'attention_mask' and optionally 'labels'
        """

        # Extract input_ids and labels
        input_ids = [example["input_ids"] for example in examples]
        labels = [example["labels"] for example in examples]

        # Pad input sequences
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

        # Pad labels if needed
        if not self.predict_with_generate:
            labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Create attention mask based on padded input
        attention_mask = input_ids.ne(0)

        # Assemble final dict
        data_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if labels is not None:
            data_dict["labels"] = labels

        return data_dict

