from dataclasses import dataclass, field

@dataclass
class LoraArguments:
    lora_r: int = field(default=32, metadata={"help": "Lora R dimension."})
    lora_alpha: int = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    lora_target_modules: str = field(default="q_proj;v_proj;k_proj", metadata={"help": "Lora target modules separated by semicolon, no space."})
    lora_weight_path: str = ""
    max_memory_MB: int = field(default=40960, metadata={"help": "Free memory per gpu."})
    lora_bias: str = "none"
    q_lora: bool = False

