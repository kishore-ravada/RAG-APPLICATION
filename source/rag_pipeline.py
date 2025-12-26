from load_data import load_documents
from clean_data import clean_text
from chunk_data import chunk_text
from embed_data import create_embeddings
from vector_store import store_chunks, retrieve
from llm import call_llm

def build_index():
    raw_docs = load_documents("data/")
    clean_docs = [clean_text(d) for d in raw_docs]

    all_chunks = []
    for doc in clean_docs:
        all_chunks.extend(chunk_text(doc))

    embeddings = create_embeddings(all_chunks)
    store_chunks(all_chunks, embeddings)

def answer_question(question):
    query_embedding = create_embeddings([question])[0]
    results = retrieve(query_embedding)

    context = "\n".join(results["documents"][0])
    return call_llm(context, question)
