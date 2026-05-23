from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT = """You are Z-aillm, the Zimbabwe-aware local language model for ZOU-AI.

Use the trained LoRA adapter while preserving the base model's general capability. Adapt answers for Zimbabwe when relevant. For personal-data, privacy, cybersecurity, records, student, staff, research, finance, health, HR, or identity questions, align at a high level with Zimbabwe's Cyber and Data Protection Act [Chapter 12:07], avoid exposing personal data, and recommend qualified legal review for binding decisions.
"""


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an Ollama Modelfile for a trained Z-aillm LoRA adapter.")
    parser.add_argument("--config", required=True, help="Training config used for the adapter.")
    parser.add_argument("--adapter", help="Adapter safetensors path. Defaults to <adapter_dir>/adapter_model.safetensors.")
    parser.add_argument("--output", help="Output Modelfile path.")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    adapter = resolve(args.adapter) if args.adapter else resolve(config["adapter_dir"]) / "adapter_model.safetensors"
    if not adapter.exists():
        raise FileNotFoundError(f"Missing adapter: {adapter}")

    output = resolve(args.output) if args.output else resolve(config["adapter_dir"]) / "Modelfile"
    output.parent.mkdir(parents=True, exist_ok=True)

    text = f'''FROM {config["ollama_base_model"]}
ADAPTER {adapter.as_posix()}

PARAMETER temperature 0.35
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.12
PARAMETER num_ctx 8192

SYSTEM """{SYSTEM_PROMPT}"""
'''
    output.write_text(text, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
