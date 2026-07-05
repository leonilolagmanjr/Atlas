import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="database")

collection = client.get_or_create_collection("atlas_knowledge")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def document_exists(document_name):
    results = collection.get(where={"source": document_name})
    return len(results["ids"]) > 0


def add_document(document_name, text):
    embedding = embedding_model.encode(text).tolist()

    collection.add(
        ids=[document_name],
        embeddings=[embedding],
        documents=[text],
        metadatas=[
            {
                "source": document_name
            }
        ]
    )


def search(question, limit=3):
    embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=limit
    )

    return results["documents"][0]