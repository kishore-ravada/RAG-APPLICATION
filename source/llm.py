import ollama

def call_llm(context, question):
    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="gemma3:1b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
