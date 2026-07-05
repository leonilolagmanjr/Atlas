from pathlib import Path
from pypdf import PdfReader

KNOWLEDGE_FOLDER = Path("knowledge")


def get_documents():
    return list(KNOWLEDGE_FOLDER.glob("*.pdf"))


def read_document(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text