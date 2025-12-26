import sys
sys.path.append("source")

import gradio as gr
from rag_pipeline import build_index, answer_question


print(" Building vector index...")
build_index()
print("RAG system ready!")


def chat_fn(user_question):
    if not user_question.strip():
        return "Please ask a question."
    return answer_question(user_question)


with gr.Blocks(title="RAG Application") as demo:
    gr.Markdown("RAG Question Answering System")
    gr.Markdown("Ask questions from your documents")

    with gr.Row():
        question = gr.Textbox(
            label="Your Question",
            placeholder="Ask something from the documents...",
            lines=2
        )

    answer = gr.Textbox(
        label="Answer",
        lines=8
    )

    ask_btn = gr.Button("submit")

    ask_btn.click(
        fn=chat_fn,
        inputs=question,
        outputs=answer
    )

demo.launch()
