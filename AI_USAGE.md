# EduSaarthi AI — AI Usage & Fallback Specification

## RAG & AI Tutor Pipeline

1. **Retrieval**: User question is searched against local markdown knowledge base (`data/knowledge_base/`) using Top-K vector term matching.
2. **Context Grounding**: Top-K retrieved sections are injected into prompt as factual data.
3. **Structured Response**: AI Tutor formats output with Explanation, Code Example, Common Misconception, Practice Question, and Grounded Sources.

## Demo AI Fallback Mode

The application will **NEVER** crash due to missing API keys or Gemini timeouts.

If `GEMINI_API_KEY` is omitted from `.env` or an API call fails:
- System activates **DEMO AI MODE**.
- Returns high-quality deterministic responses, 5-question quizzes, and 5-day intervention roadmaps.
- Displays banner: `DEMO AI MODE ACTIVE`.
