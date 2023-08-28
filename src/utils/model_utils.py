import argparse
import os
from os.path import exists, isdir, join
from typing import Any, Dict, List, Tuple

import bitsandbytes as bnb
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def add_special_tokens_if_missing(
    tokenizer: PreTrainedTokenizer, model: PreTrainedModel
) -> None:
    """
    If 'llama' or 'baichuan' is in the model name or path, check if the special tokens are set correctly.
    Add any missing special tokens to prevent them from being parsed into different tokens.
    Note that these special tokens are present in the vocabulary.
    Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.

    Args:
        tokenizer: The pre-trained tokenizer.
        model: The pre-trained model.

    Returns:
        None.
    """
    # Define a dictionary to store any missing special tokens along with their default values
    special_tokens_dict: Dict[str, Any] = {}

    # Check if each special token is present. If not, add it to the special_tokens_dict with its default value.
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # If there are any missing special tokens, call `smart_tokenizer_and_embedding_resize()` to add them to the
    # tokenizer and resize the embedding accordingly.
    if len(special_tokens_dict) > 0:
        smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict[str, str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
) -> None:
    """Resize tokenizer and embedding to accommodate new special tokens.
    Args:
        special_tokens_dict (Dict[str, str]): A dictionary of special tokens to be added to the tokenizer.
        tokenizer (PreTrainedTokenizer): The tokenizer object to be resized.
        model (PreTrainedModel): The model object whose token embeddings are to be resized.

    Returns:
        None

    Note: This function resizes the tokenizer to accommodate additional special tokens and the
    embedding matrix of the model to match the new size of the tokenizer. If any new special tokens
    have been added, the function computes the average embedding values of the existing embeddings
    and sets those values for the new special token embeddings. This is done separately for the input
    embeddings and output embeddings of the model.
    """

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        # Compute average embeddings of existing tokens
        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg



def get_last_checkpoint(checkpoint_dir: str) -> Tuple[str, bool]:
    """
    Given a directory containing previous saved checkpoints, returns the path to the last checkpoint
    if available along with a boolean flag indicating whether training has already been completed.

    Args:
        checkpoint_dir (str): Path to the directory containing the saved checkpoints.

    Returns:
        A tuple containing the path to the last checkpoint if available, and a boolean flag indicating
        whether training has already been completed.
    """
    # Check if provided directory exists
    if isdir(checkpoint_dir):
        # Check if 'completed' file exists in the directory - indicates training has completed
        is_completed = exists(join(checkpoint_dir, "completed"))
        if is_completed:
            return None, True  # Already finished

        # Find the latest checkpoint by checking all subdirectories named 'checkpoint-*'
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith(
                "checkpoint"
            ):
                max_step = max(max_step, int(filename.replace("checkpoint-", "")))
        if max_step == 0:
            return None, is_completed  # Training started, but no checkpoint found

        # Return path to the latest checkpoint directory
        checkpoint_dir = join(checkpoint_dir, f"checkpoint-{max_step}")
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed

    # The directory does not exist, meaning this is the first time the training is being run
    return None, False


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


# Avoid runtime error in model.generate(do_sample=True).
class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 0] = 1.0
        return scores


def get_logits_processor() -> LogitsProcessorList:
    logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    return logits_processor
