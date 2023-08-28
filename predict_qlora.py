import os
import json
import time
import argparse

from tqdm import trange
import torch
from transformers import GenerationConfig, HfArgumentParser

from src.configs import (
    DataArguments,
    GenerationArguments,
    LoraArguments,
    ModelArguments,
    QuantArguments,
    TrainingArguments,
)
from src.model import get_accelerate_model
from src.data import make_prediction_dataset
from src.utils.predict import generate_output



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--peft_ckpt_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="lcquad")
    parser.add_argument("--output_name", type=str, default="data/webnlg_preds_{model_name_or_path}_{time_now}.json")

    return parser.parse_args()


def predict():
    # parameters
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            TrainingArguments,
            LoraArguments,
            QuantArguments,
            GenerationArguments,
        )
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        quant_args,
        generation_args,
    ) = parser.parse_args_into_dataclasses()

    data_args.init_for_training()
    model_args.init_for_training()
    generation_config = GenerationConfig(**vars(generation_args))

    args = argparse.Namespace(
        **vars(model_args),
        **vars(data_args),
        **vars(training_args),
        **vars(lora_args),
        **vars(quant_args),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = get_accelerate_model(args, local_parser.peft_ckpt_path, do_train=False)
    model.config.use_cache = False

    dataset, dataset_labels = make_prediction_dataset(tokenizer=tokenizer, args=args)

    results = {"inputs": [], "outputs": [], "preds": []}
    batchsize = 1
    batch_num = len(dataset) // batchsize + int(len(dataset) % batchsize > 0)
    batch_num = 5
    print(f"Predict examples len: {len(dataset)}; batch_num {batch_num}\n")
    for j in trange(batch_num):
        inputs = dataset[j*batchsize : (j+1)*batchsize]
        labels = dataset_labels[j*batchsize : (j+1)*batchsize]
        outputs = generate_output(model, tokenizer, inputs, generation_config, args.model_params.stop_str)
        results['inputs'].extend(inputs)
        results['outputs'].extend(labels)
        results['preds'].extend(outputs)
    return results


if __name__ == "__main__":
    local_parser = get_args()
    result = predict()
    formats = {"model_name_or_path": local_parser.model_name_or_path.replace("/", "_"), "time_now": time.strftime('%Y-%m-%d-%H:%M')}
    out_path = local_parser.output_name
    if "{" in out_path and all([f in out_path for f in formats]):
        out_path = out_path.format(**formats)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path)

    with open(out_path, "w") as f:
        json.dump(result, f, indent = 4)
