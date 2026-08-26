import os
import glob
import math
from typing import List, Dict

class EducationalVectorStore:
    def __init__(self, kb_dir: str = None):
        if not kb_dir:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            kb_dir = os.path.join(base_dir, "..", "data", "knowledge_base")
            if not os.path.exists(kb_dir):
                kb_dir = os.path.join(base_dir, "data", "knowledge_base")
        
        self.kb_dir = kb_dir
        self.documents: List[Dict[str, str]] = []
        self._load_and_chunk_documents()

    def _load_and_chunk_documents(self):
        if not os.path.exists(self.kb_dir):
            return
            
        md_files = glob.glob(os.path.join(self.kb_dir, "*.md"))
        for file_path in md_files:
            filename = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Simple section chunking by headers
            sections = content.split("## ")
            title = sections[0].replace("# ", "").strip() if sections else filename

            for sec in sections[1:]:
                lines = sec.strip().split("\n")
                header = lines[0].strip()
                body = "\n".join(lines[1:]).strip()
                if body:
                    self.documents.append({
                        "filename": filename,
                        "title": f"{title} - {header}",
                        "text": f"{header}\n{body}"
                    })

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, str]]:
        if not self.documents:
            return []

        query_words = set(query.lower().split())
        scored_docs = []

        for doc in self.documents:
            doc_words = set(doc["text"].lower().split())
            intersection = query_words.intersection(doc_words)
            if intersection:
                # Basic TF-IDF / term-overlap score
                score = len(intersection) / (math.log(len(doc_words) + 1) + 1.0)
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_docs[:top_k]]

vector_store = EducationalVectorStore()
