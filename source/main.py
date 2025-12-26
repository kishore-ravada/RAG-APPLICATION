from rag_pipeline import build_index, answer_question

build_index()

while True:
    q = input("Ask a question: ")
    print(answer_question(q))

