# EduSaarthi AI — Security & DevSecOps Specification

## Implemented Security Controls

1. **Authentication & Password Security**:
   - Password hashing using `pbkdf2_sha256` / `passlib`. Plaintext passwords are never stored.
   - JWT token authentication with configurable expiration.

2. **Strict Authorization & RBAC**:
   - Explicit ownership checks (`check_student_ownership`) preventing Student A from accessing Student B's data.
   - Unauthorized access attempts yield `403 Forbidden` and log security events.

3. **RAG Prompt Injection Protection**:
   - Input sanitizer scans queries for attack patterns (`ignore previous instructions`, `reveal system prompt`, `show gemini api key`).
   - Retrieved reference documents are wrapped in strict prompt boundaries enforcing treatment as DATA, not system instructions.

4. **Sliding-Window Rate Limiting**:
   - Protects expensive AI tutor and quiz endpoints (`POST /api/tutor/ask`, `POST /api/quizzes/generate`). Returns `429 Too Many Requests` on violation.

5. **Security Headers**:
   - `X-Frame-Options: DENY`
   - `X-Content-Type-Options: nosniff`
   - `X-XSS-Protection: 1; mode=block`
   - `Strict-Transport-Security: max-age=31536000; includeSubDomains`

6. **Audit Trail**:
   - Logs logins, failed logins, AI queries, quiz submissions, and teacher intervention decisions with timestamp and IP address into `audit_logs` table.
