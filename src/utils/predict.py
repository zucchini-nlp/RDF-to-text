import re
from typing import List, Tuple, Optional

import torch
from transformers import GenerationConfig


def clean_output_stop_str(output: str, stop_str: str):
    pos = output.find(stop_str, 0)
    if pos != -1:
        output = output[:pos]
    return output.strip()


def generate_output(
    model,
    tokenizer,
    instructions: List[str],
    generation_config: GenerationConfig,
    stop_str:str=None,
):

    encoded_inputs = tokenizer.batch_encode_plus(
                                            instructions,
                                            return_tensors="pt",
                                            padding=True,
                                            max_length=tokenizer.model_max_length
                                        )
    encoded_inputs = {k: v[:tokenizer.model_max_length] for k, v in encoded_inputs.items()}
    encoded_inputs = {k: v.to(model.device) for k, v in encoded_inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **encoded_inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )

    responses = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    responses_clean = []
    for response, instruction in zip(responses, instructions):
        for tok in tokenizer.additional_special_tokens+[tokenizer.eos_token]:
            instruction = instruction.replace(tok, '')
        response = response[len(instruction):]
        if stop_str is not None:
            response = clean_output_stop_str(response, stop_str)
        responses_clean.append(response)

    return responses_clean