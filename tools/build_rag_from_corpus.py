from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "z-aillm" / "rag-corpus"
RAG_INDEX_FILE = ROOT / "extensions" / "ai-chat" / "data" / "rag_index.json"
SETTINGS_FILE = ROOT / "extensions" / "ai-chat" / "data" / "settings.json"
SUPPORTED_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".html", ".htm",
    ".py", ".js", ".ts", ".css", ".sql", ".log",
}


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    return {}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'-]{2,}", text.lower())


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding, errors="ignore")
        except Exception:
            continue
    return ""


def chunk_text(text: str, source: str, chunk_chars: int, overlap_chars: int) -> list[dict]:
    clean = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    index = 0
    while start < len(clean):
        end = min(start + chunk_chars, len(clean))
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append({
                "source": source,
                "chunk": index,
                "text": chunk,
                "tokens": dict(Counter(tokenize(chunk))),
            })
        if end == len(clean):
            break
        start = max(0, end - overlap_chars)
        index += 1
    return chunks


def source_key(source: str) -> str:
    try:
        return str(Path(source).expanduser().resolve())
    except Exception:
        return source.strip()


def load_existing_chunks() -> list[dict]:
    if not RAG_INDEX_FILE.exists():
        return []
    try:
        return json.loads(RAG_INDEX_FILE.read_text(encoding="utf-8")).get("chunks", [])
    except Exception:
        return []


def main() -> None:
    settings = load_settings()
    chunk_chars = max(300, int(settings.get("rag_chunk_chars", 1200)))
    overlap_chars = max(0, min(int(settings.get("rag_overlap_chars", 160)), chunk_chars // 2))

    files = [
        path for path in CORPUS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS
    ]
    new_chunks = []
    for path in files:
        text = read_text(path)
        if text:
            new_chunks.extend(chunk_text(text, str(path), chunk_chars, overlap_chars))

    existing = load_existing_chunks()
    new_sources = {source_key(chunk.get("source", "")) for chunk in new_chunks}
    preserved = [chunk for chunk in existing if source_key(chunk.get("source", "")) not in new_sources]
    merged = [*preserved, *new_chunks]

    RAG_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAG_INDEX_FILE.write_text(json.dumps({
        "chunks": merged,
        "updated_at": time.time(),
    }, indent=2), encoding="utf-8")

    print(f"Indexed {len(new_chunks)} chunks from {len(files)} corpus files.")
    print(f"Preserved {len(preserved)} existing chunks.")
    print(f"Total chunks: {len(merged)}")


if __name__ == "__main__":
    main()
