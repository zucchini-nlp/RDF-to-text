import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class DatasetAttr(object):
    dataset_name: Optional[str] = None
    hf_hub_url: Optional[str] = None
    local_filename: Optional[str] = None
    dataset_format: Optional[str] = None
    load_from_local: bool = False

    def __repr__(self) -> str:
        rep = (
            f"dataset_name: {self.dataset_name} || "
            f"hf_hub_url: {self.hf_hub_url} || "
            f"local_filename: {self.local_filename} \n"
            f"data_format: {self.dataset_format}  || "
            f"load_from_local: {self.load_from_local}"
        )
        return rep



@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default="lcquad",
        metadata={"help": "Which dataset to finetune on. See datamodule for options."},
    )

    dataset_dir: str = field(
        default=None,
        metadata={"help": "where is dataset in local dir. See datamodule for options."},
    )
    instruction_template: str = field(
        default="zeroshot_triplets",
        metadata={
            "help": "Which template to use for constructing prompts in training and inference."
        },
    )
    eval_dataset_size: Optional[float] = field(
        default=0.1, metadata={"help": "Size of validation dataset."}
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    target_max_len: int = field(
        default=256,
        metadata={
            "help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    dataset: str = field(
        default="lcquad",
        metadata={"help": "Which dataset to finetune on. See datamodule for options."},
    )
    dataset_format: Optional[str] = field(
        default="lcquad",
        metadata={
            "help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"
        },
    )

    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    output_name: str = field(
        default="data/preds_{model_name_or_path}_{time_now}.json",
        metadata={
            "help": "The path where to save the predictions in the predict script."
        },
    )

    def init_for_training(self):  # support mixing multiple datasets
        dataset_names = [ds.strip() for ds in self.dataset_name.split(",")]
        this_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(this_dir, "../..", "data")
        datasets_info_path = os.path.join(data_path, "data_info.yaml")
        with open(datasets_info_path, "r") as f:
            datasets_info = yaml.safe_load(f)

        self.datasets_list: List[DatasetAttr] = []
        for i, name in enumerate(dataset_names):
            if name not in datasets_info:
                raise ValueError(
                    "Undefined dataset {} in {}".format(name, datasets_info_path)
                )

            dataset_attr = DatasetAttr()
            dataset_attr.dataset_name = name
            dataset_attr.dataset_format = datasets_info[name].get(
                "dataset_format", None
            )
            dataset_attr.hf_hub_url = datasets_info[name].get("hf_hub_url", None)
            dataset_attr.local_filename = datasets_info[name].get("local_filename", None)

            print(os.path.join(data_path, datasets_info[name]["local_filename"]))
            if datasets_info[name]["local_filename"] and os.path.exists(
                os.path.join(data_path, datasets_info[name]["local_filename"])
            ):
                dataset_attr.load_from_local = True
            elif datasets_info[name]["local_filename"]:
                dataset_attr.load_from_local = False
                raise Warning(
                    "You have set local_path for {} but it does not exist! Will load the data from {}".format(
                        name, dataset_attr.hf_hub_url
                    )
                )

            self.datasets_list.append(dataset_attr)
