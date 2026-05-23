from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def torch_dtype(name: str):
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return "auto"


def format_example(example: dict, tokenizer) -> dict:
    messages = example.get("messages") or []
    if messages:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        text = example.get("text", "")
    return {"text": text}


def create_sft_config(config: dict, output_dir: Path) -> SFTConfig:
    from trl import SFTConfig

    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(config.get("epochs", 1)),
        "learning_rate": float(config.get("learning_rate", 2e-4)),
        "per_device_train_batch_size": int(config.get("batch_size", 1)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 8)),
        "warmup_ratio": float(config.get("warmup_ratio", 0.03)),
        "logging_steps": int(config.get("logging_steps", 5)),
        "save_steps": int(config.get("save_steps", 50)),
        "save_total_limit": 2,
        "dataset_text_field": "text",
        "packing": False,
        "bf16": config.get("torch_dtype", "bfloat16") == "bfloat16",
        "fp16": config.get("torch_dtype", "bfloat16") == "float16",
        "report_to": [],
    }

    signature = inspect.signature(SFTConfig)
    if "max_seq_length" in signature.parameters:
        kwargs["max_seq_length"] = int(config.get("max_seq_length", 2048))
    elif "max_length" in signature.parameters:
        kwargs["max_length"] = int(config.get("max_seq_length", 2048))

    supported = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return SFTConfig(**supported)


def create_sft_trainer(model, training_args, train_dataset, tokenizer, peft_config):
    from trl import SFTTrainer

    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
    }

    signature = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer

    return SFTTrainer(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Z-aillm LoRA adapter with TRL + PEFT.")
    parser.add_argument("--config", required=True, help="Path to a training YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and dataset without loading the model.")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = resolve(config["dataset"])
    output_dir = resolve(config["output_dir"])
    adapter_dir = resolve(config["adapter_dir"])

    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")

    if args.dry_run:
        examples = sum(1 for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip())
        print(json.dumps({
            "name": config["name"],
            "hf_base_model": config["hf_base_model"],
            "ollama_base_model": config["ollama_base_model"],
            "dataset": str(dataset_path),
            "examples": examples,
            "output_dir": str(output_dir),
            "adapter_dir": str(adapter_dir),
            "load_in_4bit": bool(config.get("load_in_4bit", True)),
        }, indent=2))
        return

    try:
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing training dependency: {exc.name}\n"
            "Install the LoRA training stack first:\n"
            "  .\\venv\\Scripts\\python.exe -m pip install -r .\\z-aillm\\training\\requirements.txt"
        ) from exc

    raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    if len(raw_dataset) == 0:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(config["hf_base_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = raw_dataset.map(lambda example: format_example(example, tokenizer), remove_columns=raw_dataset.column_names)

    quantization_config = None
    if bool(config.get("load_in_4bit", True)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype(config.get("torch_dtype", "bfloat16")),
        )

    model = AutoModelForCausalLM.from_pretrained(
        config["hf_base_model"],
        quantization_config=quantization_config,
        torch_dtype=torch_dtype(config.get("torch_dtype", "bfloat16")),
        device_map="auto",
    )
    model.config.use_cache = False
    if quantization_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        )

    lora = config["lora"]
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = create_sft_config(config, output_dir)

    trainer = create_sft_trainer(model, training_args, train_dataset, tokenizer, peft_config)
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
