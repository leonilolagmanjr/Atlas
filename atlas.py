import ollama

from knowledge_search import get_documents, read_document
from vector_store import document_exists, add_document, search

print("Starting Atlas...\n")

for document in get_documents():

    if document_exists(document.name):
        continue

    print(f"Learning {document.name}")

    text = read_document(document)

    add_document(document.name, text)

print("\nKnowledge ready.\n")

while True:

    question = input("Atlas > ")

    if question.lower() in ["exit", "quit"]:
        break

    knowledge = "\n\n".join(search(question))

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": "You are Atlas. Answer ONLY from the supplied knowledge. If it is not present, say you don't know."
            },
            {
                "role": "user",
                "content": f"""
Knowledge:

{knowledge}

Question:

{question}
"""
            }
        ]
    )

    print("\nAtlas:")
    print(response["message"]["content"])
    print()