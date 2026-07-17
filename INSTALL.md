# Installation

## 1) Prerequisites

- Python 3.10+ (the project uses modern typing and dataclasses)
- Ollama installed and running

## 2) Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

## 3) Install Python dependencies

The code imports these third-party libraries:

- `chromadb`
- `sentence-transformers`
- `pypdf`
- `ollama`

Install them with pip:

```powershell
pip install chromadb sentence-transformers pypdf ollama
```

## 4) Pull the configured Ollama model

By default, Atlas uses the model configured in `config.py`:

- `OLLAMA_MODEL` (default: `qwen2.5:7b`)

Pull it:

```powershell
ollama pull qwen2.5:7b
```

## 5) Prepare knowledge

Place one or more PDF files in:

- `knowledge/`

At startup, Atlas will index any changed files in that folder.

