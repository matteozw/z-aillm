from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request


ROOT = Path(__file__).resolve().parents[2]


def chat(model: str, prompt: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode("utf-8")
    req = request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("message", {}).get("content", "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an Ollama model against a small prompt set.")
    parser.add_argument("--model", default="z-aillm:latest")
    parser.add_argument("--prompts", default="z-aillm/training/eval_prompts.json")
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    if not prompts_path.is_absolute():
        prompts_path = ROOT / prompts_path
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))

    for index, prompt in enumerate(prompts, 1):
        print(f"\n## Prompt {index}\n{prompt}\n")
        print(chat(args.model, prompt))


if __name__ == "__main__":
    main()
