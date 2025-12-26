import os
from docx import Document
from pypdf import PdfReader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def load_docx(file_path):
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_documents(folder_path=DATA_DIR):
    documents = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.endswith(".txt"):
            documents.append(load_txt(file_path))

        elif file.endswith(".pdf"):
            documents.append(load_pdf(file_path))

        elif file.endswith(".docx"):
            documents.append(load_docx(file_path))

    print(f"Loaded {len(documents)} documents")
    return documents
