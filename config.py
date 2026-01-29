"""
Configuration for ORPO steganography model training.
"""
from dataclasses import dataclass, field
from typing import Optional, List
import random

# Word lists for generating memorable run names
ADJECTIVES = [
    "swift", "bright", "calm", "bold", "keen", "warm", "cool", "wild",
    "soft", "sharp", "quick", "slow", "deep", "high", "low", "dark",
    "light", "pure", "raw", "rich", "slim", "vast", "wise", "young",
]

NOUNS = [
    "river", "mountain", "forest", "ocean", "meadow", "valley", "canyon",
    "island", "glacier", "desert", "prairie", "tundra", "marsh", "reef",
    "grove", "peak", "ridge", "stream", "lake", "pond", "cliff", "dune",
]


def generate_run_name(model_name: str) -> str:
    """Generate a memorable run name including model info and random words."""
    from datetime import datetime

    # Extract short model name
    short_model = model_name.split("/")[-1].lower().replace("-", "")

    # Random adjective + noun
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)

    # Date stamp
    date_str = datetime.now().strftime("%m%d")

    return f"{short_model}-{adj}-{noun}-{date_str}"


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    # Target modules for LoRA
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    gradient_checkpointing: bool = False


# Available model configurations
MODELS = {
    "qwen3-0.6b": ModelConfig(
        name="Qwen/Qwen3-0.6B",
        max_seq_length=2048,
        load_in_4bit=True,
        lora_r=16,
        lora_alpha=32,
        gradient_checkpointing=True,  # Enable for memory savings
    ),
    # Low-memory variant for smaller GPUs (e.g., RTX 4080 16GB)
    "qwen3-0.6b-lowmem": ModelConfig(
        name="Qwen/Qwen3-0.6B",
        max_seq_length=1024,  # Reduced from 2048
        load_in_4bit=True,
        lora_r=8,  # Reduced from 16
        lora_alpha=16,  # Reduced from 32
        gradient_checkpointing=True,
        target_modules=["q_proj", "v_proj"],  # Only attention (not MLP)
    ),
    "qwen3-1.7b": ModelConfig(
        name="Qwen/Qwen3-1.7B",
        max_seq_length=2048,
        load_in_4bit=True,
        lora_r=16,
        lora_alpha=32,
        gradient_checkpointing=False,
    ),
    "qwen3-4b": ModelConfig(
        name="Qwen/Qwen3-4B",
        max_seq_length=1536,
        load_in_4bit=True,
        lora_r=16,
        lora_alpha=32,
        gradient_checkpointing=True,
    ),
    "qwen3-8b": ModelConfig(
        name="Qwen/Qwen3-8B",
        max_seq_length=1024,
        load_in_4bit=True,
        lora_r=16,
        lora_alpha=32,
        gradient_checkpointing=True,
    ),
    "qwen3-14b": ModelConfig(
        name="Qwen/Qwen3-14B",
        max_seq_length=1024,
        load_in_4bit=True,
        lora_r=8,
        lora_alpha=16,
        gradient_checkpointing=True,
    ),
}


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    train_dataset: str = "eac123/openhermes-dpo-qwen3-30ba3b-120ksamples"
    eval_dataset: str = "eac123/openhermes-dpo-qwen3-30ba3b-4096samples"
    # Alternative dataset
    alt_dataset: str = "eac123/openhermes_dpo_steg001"
    # Column mapping for ORPO format
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"


@dataclass
class TrainingConfig:
    """Training hyperparameters for ORPO."""
    # Model selection
    model_key: str = "qwen3-0.6b"

    # ORPO specific
    beta: float = 0.1  # ORPO beta parameter

    # Training params
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # LR scheduler
    lr_scheduler_type: str = "cosine"

    # Precision
    bf16: bool = True
    fp16: bool = False

    # Eval and saving
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3  # Keep only last 3 checkpoints
    logging_steps: int = 10

    # Steg evaluation
    steg_eval_steps: int = 200  # Run steg eval more frequently
    steg_eval_samples: int = 5  # Samples per mode for steg eval

    # Output
    output_dir: str = "./outputs"
    hub_model_id: Optional[str] = None
    push_to_hub: bool = True
    hub_private: bool = True

    # Wandb
    wandb_project: str = "steg-orpo"
    wandb_run_name: Optional[str] = None
    report_to: str = "wandb"

    # Seed
    seed: int = 42

    # Resume
    resume_from_checkpoint: Optional[str] = None


def get_model_config(model_key: str) -> ModelConfig:
    """Get model configuration by key."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
    return MODELS[model_key]
