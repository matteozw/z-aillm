from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "z-aillm" / "rag-corpus"
CONVERSATIONS_FILE = ROOT / "extensions" / "ai-chat" / "data" / "conversations.json"
OUTPUT_FILE = ROOT / "z-aillm" / "fine-tuning" / "datasets" / "z-aillm-instructions.jsonl"
SYSTEM_PROMPT = (
    "You are Z-aillm, a Zimbabwe-aware assistant. Preserve general Llama capability, "
    "adapt answers for Zimbabwe when relevant, and give high-level Cyber and Data Protection "
    "Act [Chapter 12:07] guidance without pretending to be a lawyer."
)


def clean(text: str, limit: int = 3500) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def messages(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": clean(user, 1800)},
            {"role": "assistant", "content": clean(assistant, 2500)},
        ]
    }


def seed_examples() -> list[dict]:
    return [
        messages(
            "Hello",
            "Hello. I am Z-aillm, ready to help with general questions and Zimbabwe-aware guidance.",
        ),
        messages(
            "What is the safest way to answer questions about student personal data?",
            "Give general guidance, avoid exposing real personal data, recommend data minimisation and access controls, and say that binding decisions under Zimbabwe's Cyber and Data Protection Act [Chapter 12:07] need qualified legal review.",
        ),
        messages(
            "Draft a short privacy notice for a Zimbabwean university helpdesk.",
            "This helpdesk collects the information needed to respond to your request, such as your name, contact details, student or staff reference where required, and the details of the issue. The information is used only for support, record keeping, service improvement, and lawful institutional purposes. Access is limited to authorised staff, and records are retained only as long as necessary under approved policy. Do not submit unnecessary sensitive information.",
        ),
    ]


def examples_from_corpus() -> list[dict]:
    examples = []
    for path in sorted(CORPUS_DIR.rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        text = clean(path.read_text(encoding="utf-8", errors="ignore"), 3000)
        if text:
            examples.append(messages(
                f"Summarise the key points from this Zimbabwe reference note: {path.name}",
                f"Key points: {text}",
            ))
    return examples


def examples_from_conversations() -> list[dict]:
    if not CONVERSATIONS_FILE.exists():
        return []
    try:
        payload = json.loads(CONVERSATIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

    conversations = payload.get("conversations", [])
    examples = []
    for conversation in conversations:
        for turn in conversation.get("history", []):
            if not isinstance(turn, list | tuple) or len(turn) < 2:
                continue
            user, assistant = clean(turn[0], 1800), clean(turn[1], 2500)
            if user and assistant and "Ollama is not reachable" not in assistant:
                examples.append(messages(user, assistant))
    return examples


def main() -> None:
    examples = [*seed_examples(), *examples_from_corpus(), *examples_from_conversations()]
    seen = set()
    unique = []
    for example in examples:
        key = json.dumps(example, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(example)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        for example in unique:
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")

    print(f"Wrote {len(unique)} examples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
