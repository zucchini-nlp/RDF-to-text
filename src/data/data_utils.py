import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


instruction_with_q = """
A chat between a curious human and an artificial intelligence assistant.
The assistant's job is to answer the given question using only the information provided in the RDF triplet format. The assistant's answer should be in a human-readable format, with proper sentences and grammar and should be concise and short.
The RDF triplets will be provided in triplets, where triplets are always in the (subject, relation, object) format and are separated by a semicolon. The assistant should understand that if multiple triplets are provided, the answer to the question should use all of the information from triplets and make aggregation. The assistant MUST NOT add any additional information, beside form the one proveded in the triplets.
The assistant should try to reply as short as possible, and perform counting or aggregation operations over triplets by himself when necessary.
"""

instruction_wo_q = """
A chat between a curious human and an artificial intelligence assistant.
The assistant's job is convert the provided input in RDF triplet format into human-readable text format, with proper sentences and grammar. The triplets are always in the (subject, relation, object) format, where each triplet is separated by a semicolon. The assistant should understand that if multiple triplets are provided, the generated human-readable text should use all of the information from input. The assistant MUST NOT add any additional information, beside form the one proveded in the input.
"""

instruction_zero_shot_wo_q = """
Rewrite the following triplets to human-readable full sentence in natural language.
Triplets: 
"""

instruction_zero_shot_with_q = """
Generate long sentence answer to the following question using the provided RDF triplets.
"""

history_with_q = [
        ("Human", "Question: Is Essex the Ceremonial County of West Tilbury? Triplets: ('West Tilbury', 'Ceremonial County', 'Essex')"),
        ("Assistant", "Essex is the Ceremonial County of West Tributary"),
        ("Human", "Question: What nation is Hornito located in, where Jamie Bateman Cayn died too? Triplets: ('Jaime Bateman Cayón', 'death place', 'Panama'); ('Hornito, Chiriquí', 'country', 'Panama')"),
        ("Assistant", "Hornito, Chiriquí is located in Panama, where Jaime Bateman Cayón died."),
        ("Human", "Question: Who are the shareholder of the soccer club for whom Steve Holland plays? Triplets: ('Steve Holland', 'current club', 'Chelsea F.C.'); ('Chelsea F.C.', 'owner', 'Roman Abramovich')"),
        ("Assistant", "Roman Abramovich owns Chelsea F.C., where Steve Holland plays."),
        ("Human", "Question: Who is the chancellor of Falmouth University? Triplets: ('Falmouth University', 'chancellor', 'Dawn French')"),
        ("Assistant", "The chancellor of the Falmouth University is Dawn French.")

    ]

history_wo_q = [
        ("Human", "('West Tilbury', 'Ceremonial County', 'Essex')"),
        ("Assistant", "Essex is the Ceremonial County of West Tributary"),
        ("Human", "('Jaime Bateman Cayón', 'death place', 'Panama'); ('Hornito, Chiriquí', 'country', 'Panama')"),
        ("Assistant", "Hornito, Chiriquí is located in Panama, where Jaime Bateman Cayón died."),
        ("Human", "('Steve Holland', 'current club', 'Chelsea F.C.'); ('Chelsea F.C.', 'owner', 'Roman Abramovich')"),
        ("Assistant", "Roman Abramovich owns Chelsea F.C., where Steve Holland plays."),
        ("Human", "('Falmouth University', 'chancellor', 'Dawn French')"),
        ("Assistant", "The chancellor of the Falmouth University is Dawn French.")
    ]


mapping = {
    "fewshot_question": (instruction_with_q, history_with_q),
    "fewshot_triplets": (instruction_wo_q, history_wo_q),
    "zeroshot_question": (instruction_zero_shot_with_q, []),
    "zeroshot_triplets": (instruction_zero_shot_wo_q, [])
}


def prepare_input(example: Dict[str, Any], style: str) ->  str:
    linearized_triplet = example["input"]
    question = example['question']
    if "question" in style:
        input_text = f"Question: {question.strip()} Triplets: {linearized_triplet}"
    else:
        input_text = linearized_triplet
    return {"input": input_text}


def make_prompt(
    example: Dict[str, Any],
    instruction: str,
    roles: List[str],
    history: List[Tuple[str, str]]=None,
    sep_toks: List[str]=None,
    style: str="vicuna"
):
    curr_input = example["input"]
    
    # zero-shot style
    if not history:
        ret = f"{instruction}{curr_input}\nResponse: "
    elif "vicuna" in style:
        sep_tok = sep_toks[0]
        ret = instruction + sep_tok
        for i, (role, message) in enumerate(history):
            ret += roles[i % 2] + ": " + message + sep_tok
        ret += roles[0] + ": " + curr_input + sep_tok + roles[1] + ": \n"
    elif "pythia" in style:
        sep_tok = sep_toks[0]
        ret = instruction
        for i, (role, message) in enumerate(history):
            ret += roles[i % 2] + message + sep_tok
        ret += roles[0] + ": " + curr_input + sep_tok + roles[1] + ": "
    else:
        ret = instruction
        for i, (role, message) in enumerate(history):
            ret += roles[i % 2] + message + "\n"
        ret += roles[0] + ": " + curr_input + "\n" + roles[1] + ": "
    example["input"] = ret
    return example


def local_dataset(
    dataset_path: str, eval_dataset_size: float = 0.1
) -> Tuple[Dataset, Dataset]:
    """
    Reads in a dataset from a file and returns it as a split train-test dataset.
    """

    # Read in the full dataset from file based on the file format
    if dataset_path.endswith(".json"):
        full_dataset = load_dataset("json", data_files=dataset_path)
    elif dataset_path.endswith(".jsonl"):
        full_dataset = load_dataset("json", data_files=dataset_path)
    elif dataset_path.endswith(".csv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_path))
    elif dataset_path.endswith(".tsv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_path, delimiter="\t"))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    if "train" not in full_dataset:
        split_dataset = full_dataset.train_test_split(test_size=eval_dataset_size)
        return split_dataset
    else:
        return full_dataset


def load_data(
    dataset_path: str, cache_dir="~/.cache/.huggingface", eval_dataset_size: float = 0.1
) -> Union[Dict[str, Dataset], None]:
    """
    Load a dataset based on its name.
    """
    if isinstance(dataset_path, str) and not os.path.exists(dataset_path):
        print(f"Lodding dataset from huggingface, please ref to https://huggingface.co/datasets/{dataset_path}")
        dataset = load_dataset(dataset_path, cache_dir=cache_dir)
    elif isinstance(dataset_path, list):
        dataset = load_dataset(*dataset_path, cache_dir=cache_dir)
    else:
        try:
            print(f"Lodding dataset from local path: {dataset_path}")
            dataset = local_dataset(dataset_path, eval_dataset_size)
        except Exception as e:
            raise ValueError(f"Error {e} loading dataset from {dataset_path}")
    return dataset


def process_web_nlg(example: Dict[str, Any]):
    """
    Process the web-nlg dataset so that the inputs is a linearized triplets and the output contains only good lexicalixations.

    """
    triplets = ""
    outputs = []
    for triplet in example["original_triple_sets"]["otriple_set"][0]:
        subj, rel, obj = triplet.split("|")
        triplets += f"('{subj.strip()}', '{rel.strip()}', '{obj.strip()}');"
    for text, comment in zip(example["lex"]['text'], example["lex"]['comment']):
        if comment == "good":
            outputs.append(text)
    example['input'] = triplets
    example['output'] = outputs or [""]
    return example
    

def format_instruction_dataset(
    dataset: Dataset,
    dataset_name: str,
    model_type: str,
    dataset_format: str,
    instruction_template: str = "zeroshot_triplets",
    sep_toks: List[str] = [],
    roles: List[str] = ["User", "Assistant"],
) -> Optional[Dict[str, Dataset]]:
    """
    Formats a given dataset based on its name and format.

    Removes unused columns, renames columns to 'input' and 'output',
    and applies dataset-specific formatting based on the dataset_name.

    Returns formatted dataset dict if dataset can be formatted, else None.
    """

    def _remove_unused_columns(dataset):
        dataset = dataset.remove_columns(
            [
                col
                for col in dataset.column_names["train"]
                if col not in ["input", "output"]
            ]
        )
        return dataset

    def _remove_rows(dataset, exclude_idx):
        dataset = dataset.select(
            (
                i for i in range(len(dataset)) 
                if i not in set(exclude_idx)
            )
        )
        return dataset

    print(f"The {dataset_name} using {dataset_format} dataset format.")
    print(f"Applying instruction template: {instruction_template}")
    dataset = dataset.filter(lambda example: len(example["input"]) <= 1500)
    dataset = dataset.map(prepare_input, fn_kwargs={"style": instruction_template})
    instruction, history = mapping[instruction_template]
    dataset = dataset.map(
        make_prompt, 
        fn_kwargs={"instruction": instruction, "history": history, "roles": roles, "sep_toks": sep_toks, "style": model_type})
    dataset = _remove_unused_columns(dataset)

    return dataset


def split_train_eval(
    dataset: Dataset,
    do_eval: bool = False,
    eval_dataset_size: float = 0.1,
    max_eval_samples: int = None,
    do_train: bool = True,
    max_train_samples: int = None,
) -> Dict[str, Dataset]:
    """
    Prepare the training and evaluation datasets for a machine learning model.
    """

    if not isinstance(dataset, DatasetDict):
        raise TypeError("The 'dataset' argument must be a DatasetDict object.")

    train_dataset, eval_dataset = None, None
    if do_eval:
        if "eval" in dataset:
            eval_dataset = dataset["eval"]
        else:
            print(
                f"Splitting the dataset into train and validation according to `eval_dataset_size`:  {eval_dataset_size}"
            )
            dataset = dataset["train"].train_test_split(
                test_size=eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset["test"]

        print(
            f"You have set the max_eval_samples: {max_eval_samples}, will do sampling ..."
        )
        if max_eval_samples is not None and len(eval_dataset) > max_eval_samples:
            eval_dataset = eval_dataset.select(np.arange(max_eval_samples))

    if do_train:
        train_dataset = dataset["train"]
        print(f"You have set the max_train_samples: {max_train_samples}, will do sampling ...")
        if max_train_samples is not None and len(train_dataset) > max_train_samples:
            train_dataset = train_dataset.select(np.arange(max_train_samples))

    return train_dataset, eval_dataset


def make_data_module(args, split: bool = True):
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }
    """

    train_datasets: List[Dataset] = []
    eval_datasets: List[Dataset] = []
    dataset_name_list = args.dataset_name.split(",")
    print(f"Loading datasets: {dataset_name_list}")

    for dataset_attr in args.datasets_list:
        print("=" * 80)
        print("DatasetAttr: {}".format(dataset_attr))

        if dataset_attr.load_from_local:
            dataset_file = dataset_attr.local_filename
            this_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(this_dir, "../..", "data", dataset_file)
        elif dataset_attr.hf_hub_url:
            data_path = dataset_attr.hf_hub_url
        else:
            raise ValueError("Please set the dataset path or hf_hub_url.")

        dataset = load_data(data_path, cache_dir=args.cache_dir, eval_dataset_size=args.eval_dataset_size)
        if "web_nlg" in data_path:
            dataset = dataset.map(process_web_nlg)

        dataset = format_instruction_dataset(
            dataset,
            dataset_name=dataset_attr.dataset_name,
            model_type=args.model_params.model_type,
            dataset_format=dataset_attr.dataset_format,
            instruction_template=args.instruction_template,
            sep_toks=args.model_params.sep_toks,
            roles=args.model_params.roles
            )

        if not split:
            return dataset
        
        train_dataset, eval_dataset = split_train_eval(
            dataset,
            do_eval=args.do_eval,
            eval_dataset_size=args.eval_dataset_size,
            max_eval_samples=args.max_eval_samples,
            do_train=args.do_train,
            max_train_samples=args.max_train_samples,
        )
        if train_dataset:
            print(f"loaded dataset: {dataset_attr.dataset_name} #train data size: {len(train_dataset)}")
            train_datasets.append(train_dataset)
        if eval_dataset:
            print(f"#eval data size: {len(eval_dataset)}")
            eval_datasets.append(eval_dataset)

    concate_train = concatenate_datasets(train_datasets) if train_datasets else None
    concate_eval = concatenate_datasets(eval_datasets) if eval_datasets else None
    return concate_train, concate_eval
