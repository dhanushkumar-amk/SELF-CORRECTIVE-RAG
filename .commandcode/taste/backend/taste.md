# Backend Engineering Preferences

- Prefers explicit, documented exception-to-HTTP-status mappings via centralized handlers rather than defaulting everything to 500; returns a standard error shape without leaking raw tracebacks. Confidence: 0.9
- Prefers structured JSON logging with a per-request correlation/request ID (UUID) emitted in both logs and response headers for tracing. Confidence: 0.85
- Prefers API versioning via a /api/v1/ route prefix, even for portfolio projects, to signal awareness of API evolution. Confidence: 0.85
- Prefers self-documenting APIs: realistic example payloads on Pydantic models so the OpenAPI/Swagger /docs shows real request/response shapes. Confidence: 0.85
- Prefers keeping the public API contract (request/response Pydantic models) separate from internal domain models so the public contract is independently versionable; uses consistent PascalCase naming. Confidence: 0.8
- Prefers lightweight rate limiting on public-facing endpoints to protect external LLM/API quotas, with the chosen limit documented. Confidence: 0.8
