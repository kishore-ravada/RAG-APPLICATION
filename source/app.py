import sys
sys.path.append("source")

import gradio as gr
import pyttsx3
import soundfile as sf
from faster_whisper import WhisperModel

from rag_pipeline import build_index, answer_question


# -----------------------------
# Initialize models
# -----------------------------
print("🔧 Building vector index...")
build_index()
print("✅ RAG system ready!")

print("🔧 Loading Whisper model...")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
print("✅ Whisper loaded!")

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 170)


# -----------------------------
# Helper functions
# -----------------------------
def speech_to_text(audio_path):
    if audio_path is None:
        return ""

    segments, _ = whisper_model.transcribe(audio_path)
    text = "".join([seg.text for seg in segments])
    return text.strip()


def text_to_speech(text):
    tts_engine.say(text)
    tts_engine.runAndWait()


def chat_text(user_question):
    if not user_question.strip():
        return "Please ask a question."

    answer = answer_question(user_question)
    text_to_speech(answer)
    return answer


def chat_voice(audio):
    # Convert audio to text
    user_text = speech_to_text(audio)

    if not user_text:
        return "", "Could not understand audio."

    answer = answer_question(user_text)
    text_to_speech(answer)

    return user_text, answer


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="RAG Voice Assistant") as demo:
    gr.Markdown("## 🎙️ RAG Question Answering System")
    gr.Markdown("Ask questions using **text or voice**")

    with gr.Row():
        question = gr.Textbox(
            label="Your Question (Text)",
            placeholder="Ask something from the documents...",
            lines=2
        )

    answer = gr.Textbox(
        label="Answer",
        lines=8
    )

    ask_btn = gr.Button("Submit (Text)")

    ask_btn.click(
        fn=chat_text,
        inputs=question,
        outputs=answer
    )

    gr.Markdown("### 🎤 Voice Input")

    audio_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Speak your question"
    )

    recognized_text = gr.Textbox(
        label="Recognized Speech"
    )

    voice_btn = gr.Button("Submit (Voice)")

    voice_btn.click(
        fn=chat_voice,
        inputs=audio_input,
        outputs=[recognized_text, answer]
    )

demo.launch()
