import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection("docs")

def store_chunks(chunks, embeddings):
    for i, (chunk, embed) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[embed.tolist()],
            ids=[str(i)]
        )

def retrieve(query_embedding, top_k=3):
    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
