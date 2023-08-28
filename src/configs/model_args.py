from dataclasses import dataclass, field
from typing import Optional
import os
import yaml


@dataclass
class ModelParams(object):
    model_name_or_path: str = None
    sep_toks: Optional[str] = None
    roles: Optional[str] = None
    stop_str: str = None
    load_from_local: bool = False

    def __repr__(self) -> str:
        rep = (
            f"model_name_or_path: {self.model_name_or_path} || "
            f"sep_toks: {self.sep_toks} || "
            f"roles: {self.roles} \n"
            f"stop_str: {self.stop_str}  || "
            f"load_from_local: {load_from_local}"
        )
        return rep


@dataclass
class ModelArguments:
    cache_dir: Optional[str] = field(default="/archive/turganbay/.huggingface")
    model_name_or_path: Optional[str] = field(
        default="lmsys/vicuna-13b-v1.3",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to\
              train a model from scratch."
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."},
    )
    peft_ckpt_path: str = field(
        default=None,
        metadata={"help": "Peft checkpoint path."},
    )

    def init_for_training(self):
        name = self.model_name_or_path
        this_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(this_dir, "../..", "models")
        model_info_path = os.path.join(this_dir, "../..", "models", "model_info.yaml")
        with open(model_info_path, "r") as f:
            model_info = yaml.safe_load(f)
        
        self.model_params = ModelParams()
        if name not in model_info:
                raise ValueError(
                    "Undefined model {} in {}".format(name, model_info_path)
                )

        self.model_params.model_type = model_info[name].get("model_type", name)
        self.model_params.sep_toks = model_info[name].get("sep_toks", [])
        self.model_params.roles = model_info[name].get("roles", [])
        self.model_params.stop_str = model_info[name].get("stop_str", "")
        self.model_params.load_from_local = model_info[name].get("load_from_local", False)
        if model_info[name]["load_from_local"] and not os.path.exists(os.path.join(model_dir, name)):
            raise Warning(
                f"You have set load_from_local for {name} but it does not exist! Trying to load the model from HF hub"
            )