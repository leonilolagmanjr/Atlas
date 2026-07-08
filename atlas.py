"""Atlas V3 entry point.

Entry point only: startup, initialization, and chat loop.
"""

from __future__ import annotations

import logging
from pathlib import Path

from brain import Brain
from config import KNOWLEDGE_FOLDER, LOG_FILE, LOG_LEVEL, LOG_TO_FILE
from indexer import index_knowledge_base
from logger import setup_logging
from vector_store import VectorStore

logger = logging.getLogger(__name__)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_prompts() -> tuple[str, str]:
    prompts_dir = Path(__file__).parent / "prompts"
    system_prompt = _read_text_file(prompts_dir / "system.txt")
    retrieval_template = _read_text_file(prompts_dir / "retrieval.txt")
    return system_prompt, retrieval_template


def main() -> None:
    setup_logging(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO), log_to_file=LOG_TO_FILE and str(LOG_FILE))

    system_prompt, retrieval_template = _load_prompts()

    vector_store = VectorStore()

    logger.info("Startup: indexing knowledge base")
    index_knowledge_base(vector_store=vector_store, knowledge_folder=KNOWLEDGE_FOLDER)

    brain = Brain(
        vector_store=vector_store,
        system_prompt=system_prompt,
        retrieval_template=retrieval_template,
    )

    logger.info("Startup: entering chat loop")

    while True:
        question = input("Atlas > ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return

        print("\nAtlas:")
        print(brain.process(question))


if __name__ == "__main__":
    main()

