# Z-aillm build kit

This folder builds `z-aillm:latest` for Ollama and feeds the ZOU-AI app with a Zimbabwe-aware model option.

## 1. Build the Ollama model

```powershell
cd C:\AI\stable-diffusion-webui
ollama create z-aillm -f .\z-aillm\Modelfile
ollama run z-aillm
```

The public UI shows this as `Z-aillm`. Internally it calls `z-aillm:latest`.

## 2. RAG corpus

Put Zimbabwe-specific source documents in `z-aillm\rag-corpus`. Supported formats match the app's RAG indexer: PDF, TXT, Markdown, CSV, JSON, YAML, HTML, code, SQL, and log files.

To rebuild the app's RAG index from this corpus:

```powershell
.\venv\Scripts\python.exe .\z-aillm\tools\build_rag_from_corpus.py
```

## 3. Fine-tuning path

Ollama does not train adapters itself. This build kit uses Hugging Face Transformers, TRL, and PEFT to train LoRA adapters, then hands the adapter back to Ollama.

Install the training stack:

```powershell
.\venv\Scripts\python.exe -m pip install -r .\z-aillm\training\requirements.txt
```

Build the dataset:

```powershell
.\venv\Scripts\python.exe .\z-aillm\tools\prepare_finetune_dataset.py
```

Train a LoRA adapter:

```powershell
.\z-aillm\train-z-aillm.ps1 -Profile llama3.1-8b
```

Other supported profiles:

- `llama3.1-8b`
- `llama3.1-70b`
- `mixtral-8x7b`

After training, build an Ollama model from the adapter:

```powershell
.\z-aillm\build-ollama-adapter.ps1 -Profile llama3.1-8b -ModelName z-aillm-tuned
.\venv\Scripts\python.exe .\z-aillm\tools\eval_ollama_model.py --model z-aillm-tuned
```

Notes:

- Llama 3.1 and Mixtral Hugging Face checkpoints may require accepting model terms and logging in with `huggingface-cli login`.
- QLoRA with `bitsandbytes` is best run on Linux/WSL with a CUDA GPU. Native Windows training may need full precision or a different backend.
- `llama3.1:70b` and `mixtral:8x7b` require substantially more VRAM/RAM than `llama3.1:8b`.

Keep private personal data out of fine-tuning datasets unless you have explicit authority, a lawful purpose, minimisation, retention controls, and a deletion path.
