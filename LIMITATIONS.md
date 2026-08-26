# EduSaarthi AI — Limitations & Hackathon Constraints

1. **Database**: Uses SQLite for MVP local storage. Designed with SQLAlchemy ORM abstraction for direct migration to PostgreSQL.
2. **Rate Limiting**: In-memory sliding window rate limiter. Can be replaced with Redis for multi-instance deployments.
3. **Knowledge Base**: Demo knowledge base contains curated Python basics and DBMS / SQL JOIN materials.
4. **AI Models**: Integrates Gemini 1.5 Flash API with deterministic offline fallback mode.
