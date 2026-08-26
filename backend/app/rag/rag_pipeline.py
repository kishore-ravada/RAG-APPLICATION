from typing import Dict
from app.rag.vector_store import vector_store
from app.ai.gemini_client import get_ai_tutor_response
from app.security.sanitizer import sanitize_input, check_prompt_injection
from sqlalchemy.orm import Session

def query_rag_pipeline(question: str, level: str = "INTERMEDIATE", client_ip: str = "127.0.0.1", db: Session = None) -> Dict:
    """
    RAG Pipeline:
    1. Sanitize & check prompt injection.
    2. Search vector store for top-K educational chunks.
    3. Construct grounded prompt treating retrieved docs as DATA.
    4. Return structured response with sources.
    """
    clean_question = sanitize_input(question)
    check_prompt_injection(clean_question, client_ip=client_ip, db=db)

    retrieved_docs = vector_store.search(clean_question, top_k=2)
    sources_format = [{"title": d["title"], "text": d["text"]} for d in retrieved_docs]

    return get_ai_tutor_response(clean_question, level, sources_format)
