# Atlas v1

> A lightweight local AI assistant built on Qwen 2.5 that learns through Retrieval-Augmented Generation (RAG), user feedback, and long-term memory.

---

# Overview

Atlas is a personal AI assistant designed to run completely on local hardware using Ollama. Instead of retraining a language model every time new information is learned, Atlas stores knowledge inside a vector database and retrieves relevant information during conversations.

The goal of Atlas is to become a continually improving AI assistant capable of remembering information, learning from documentation, and adapting based on rewards and penalties.

Current Base Model:

- Qwen2.5:7B
- Ollama
- Python

---

# Features

## Current Features

- Local AI using Ollama
- Qwen2.5 7B base model
- Python backend
- Interactive CLI chat
- Retrieval-Augmented Generation (RAG)
- PDF knowledge ingestion
- Long-term memory
- Conversation history
- Learning from documentation
- Reward/Penalty learning system
- Semantic search
- Embedding-based memory retrieval

---

# Planned Features

- GUI desktop application
- Voice input/output
- Vision support
- Autonomous agents
- Tool calling
- Internet search
- Code execution
- Plugin system
- API server
- Multi-user support
- Self-improving memory organization
- Automatic summarization
- Personal assistant mode

---

# Project Structure

```
Atlas/
│
├── data/
│   ├── documents/
│   ├── embeddings/
│   └── memories/
│
├── models/
│
├── rag/
│   ├── ingest.py
│   ├── retrieve.py
│   └── embeddings.py
│
├── learning/
│   ├── rewards.py
│   ├── penalties.py
│   └── memory.py
│
├── prompts/
│
├── logs/
│
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

---

# Technology Stack

| Component | Technology |
|-----------|------------|
| AI Model | Qwen2.5:7B |
| Runtime | Ollama |
| Language | Python 3.12+ |
| Vector Database | ChromaDB |
| Embeddings | nomic-embed-text |
| PDF Parsing | PyMuPDF |
| RAG | LangChain |
| CLI | Rich |
| Memory | ChromaDB |

---

# Installation

## Clone the repository

```bash
git clone https://github.com/yourusername/Atlas.git

cd Atlas
```

---

## Create a Virtual Environment

Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

Linux/macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download:

https://ollama.com

---

## Pull the AI Model

```bash
ollama pull qwen2.5:7b
```

---

## Pull Embedding Model

```bash
ollama pull nomic-embed-text
```

---

# Running Atlas

```bash
python main.py
```

---

# Learning Workflow

```
PDF

↓

Extract Text

↓

Split into Chunks

↓

Generate Embeddings

↓

Store in ChromaDB

↓

User asks question

↓

Atlas retrieves relevant chunks

↓

Qwen2.5 generates response

↓

User rewards or penalizes answer

↓

Memory updated
```

---

# Reward System

Atlas can improve over time through feedback.

Example:

```
/reward

Great explanation.
```

Atlas increases confidence for that retrieval.

Penalty:

```
/penalty

Hallucinated information.
```

Atlas lowers confidence and records corrections.

---

# RAG Pipeline

```
Documents

↓

Chunking

↓

Embeddings

↓

Vector Database

↓

Similarity Search

↓

Context

↓

Qwen2.5

↓

Answer
```

---

# Vision

Atlas aims to become a completely local AI capable of:

- Remembering previous conversations
- Learning from books
- Reading PDFs
- Searching personal documents
- Acting as a coding assistant
- Becoming a personalized AI companion
- Running entirely offline
- Preserving user privacy

---

# Roadmap

## Version 1

- [x] Local chat
- [x] Ollama integration
- [x] RAG
- [x] PDF learning
- [x] Memory
- [x] Feedback system

---

## Version 2

- [ ] Desktop GUI
- [ ] Better memory ranking
- [ ] Automatic learning
- [ ] Voice support
- [ ] Better prompts

---

## Version 3

- [ ] Agent system
- [ ] Web search
- [ ] Code execution
- [ ] Vision
- [ ] Tool calling
- [ ] Multi-agent collaboration

---

# Philosophy

Atlas is built on one core idea:

> **Large Language Models should not memorize everything—they should know how to find the right information when needed.**

By combining local language models with Retrieval-Augmented Generation (RAG), semantic memory, and user feedback, Atlas becomes increasingly useful without requiring expensive retraining.

---

# License

MIT License

---

# Author

**Leonilo P. Lagman Jr.**

Computer Science Graduate

Software Engineer

Founder of Atlas AI

---

# Acknowledgements

Special thanks to the open-source community and the projects that make Atlas possible:

- Ollama
- Qwen
- LangChain
- ChromaDB
- PyMuPDF
- Rich
- Python

---

**Atlas v1** — *Learn locally. Remember forever. Protect your privacy.*
