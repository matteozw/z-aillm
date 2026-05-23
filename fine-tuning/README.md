# Z-aillm fine-tuning workspace

Generated instruction datasets go in `datasets`.

Recommended dataset format is JSONL with one object per line:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Use fine-tuning for durable behavior, style, and domain patterns. Use RAG for factual/legal/institutional content that changes or needs traceable sources.

Training commands:

```powershell
.\venv\Scripts\python.exe -m pip install -r .\z-aillm\training\requirements.txt
.\venv\Scripts\python.exe .\z-aillm\tools\prepare_finetune_dataset.py
.\z-aillm\train-z-aillm.ps1 -Profile llama3.1-8b
.\z-aillm\build-ollama-adapter.ps1 -Profile llama3.1-8b -ModelName z-aillm-tuned
```

Before training, review the dataset for:

- Personal data leakage
- Copyright or licensing problems
- Outdated law or policy text
- Hallucinated citations
- Examples that overstate legal certainty
