"""Credential-free keyword search over indexed PDF passages."""

import re
from difflib import SequenceMatcher
from typing import Any

from app.models.schemas import ClaimWithSource, RetrievalResult
from app.ingestion.chunk_metadata import create_vector_id
from app.retrieval.bm25_index import get_bm25_index, tokenize_text

_COMMON_WORDS = set(
    "a an and are about as at be by can could did do does for from had has have how i in "
    "is it me of on or please should the their them this to was were what when where "
    "which who why will with would you".split()
)
_BROAD_QUERY_WORDS = {"about", "describe", "overview", "summarize", "summary", "tell"}


def find_passages(query: str, document_id: str | None = None) -> dict[str, Any]:
    terms = set(tokenize_text(query)) - _COMMON_WORDS
    # Match misspellings of internship to the role label commonly used in resumes.
    internship_query = any(
        term in {"intern", "internships"} or SequenceMatcher(None, term, "internship").ratio() >= 0.7
        for term in terms if term.startswith("int")
    )
    if internship_query:
        terms.update({"intern", "internship"})
    index = get_bm25_index()
    candidates = []
    if terms:
        for match in index.search(" ".join(sorted(terms)), top_k=index.chunk_count):
            metadata = match["metadata"]
            if document_id and metadata["document_id"] != document_id:
                continue
            source = metadata["source_text"]
            if "ai" in terms and terms.intersection({"skill", "skills"}) and not terms.intersection({"explain", "compare", "missing", "improve", "why", "backend", "frontend"}):
                section = re.search(
                    r"\b(?:AI\s*(?:&|and)\s*Generative AI|AI Skills)\s*:\s*(.+?)"
                    r"(?=\s+(?:Frontend|Backend|Databases|DevOps|Education|Experience)\s*:|\n\s*\n)",
                    source, re.I | re.S,
                )
                if section and document_id:
                    skills = [item.strip() for item in section[1].split(",") if item.strip()]
                    answer = "\n".join(f"? {skill}" for skill in dict.fromkeys(skills))
                    claim = ClaimWithSource(claim_text=answer, source_chunk_id=match["id"],
                        source_text=section[0], page_number=metadata["page_number"],
                        document_id=document_id, filename=metadata.get("filename"), is_valid_source=True)
                    return {"final_status": "answer", "final_answer_text": answer, "claims": [claim],
                            "retrieved_chunks": [RetrievalResult(chunk_id=match["id"], score=match["score"], metadata=metadata)]}
            if internship_query:
                role = re.search(r"\bintern(?:ship)?\b", source, re.I)
                if role:
                    start = max(0, role.start() - 60)
                    source = re.split(r"\bProjects\s+(?=[A-Z])", source[start:], maxsplit=1)[0]
                    metadata = {**metadata, "source_text": source}
            for passage in re.split(r"(?<=[.!?])\s+|\n{2,}", source):
                passage = passage.strip()
                overlap = len(terms.intersection(tokenize_text(passage)))
                if overlap and passage:
                    candidates.append((overlap, match["score"], match["id"], passage, metadata))

    # Broad prompts such as "tell me about me" have no useful lexical terms to
    # match. Give the local model the opening chunks as document context instead
    # of returning an empty result. Keep ordinary unmatched keyword searches empty.
    if not candidates and (document_id or terms.intersection(_BROAD_QUERY_WORDS)):
        for chunk, metadata in zip(index.chunks, index.metadata_map):
            if document_id and metadata["document_id"] != document_id:
                continue
            source = str(metadata.get("source_text") or "").strip()
            if not source:
                continue
            candidates.append((
                0,
                0.0,
                create_vector_id(chunk.document_id, chunk.chunk_id),
                source,
                metadata,
            ))
            if len(candidates) == 8:
                break

    candidates.sort(key=lambda item: item[:2], reverse=True)
    claims = []
    chunks = {}
    seen = set()
    min_overlap = (candidates[0][0] + 1) // 2 if candidates else 0
    for overlap, score, chunk_id, passage, metadata in candidates:
        if overlap < min_overlap:
            break
        if passage in seen:
            continue
        seen.add(passage)
        claims.append(ClaimWithSource(
            claim_text=passage,
            source_chunk_id=chunk_id,
            source_text=metadata["source_text"],
            page_number=metadata["page_number"],
            page_number_end=metadata.get("page_number_end"),
            document_id=metadata["document_id"],
            filename=metadata.get("filename"),
            is_valid_source=True,
        ))
        chunks[chunk_id] = RetrievalResult(chunk_id=chunk_id, score=score, metadata=metadata)
        if len(claims) == (8 if overlap == 0 else 3):
            break

    return {
        "final_status": "source_passages" if claims else "unverifiable",
        "final_answer_text": "\n\n".join(claim.claim_text for claim in claims)
        or "No source passages matched this question. Try a more specific keyword from your documents.",
        "claims": claims,
        "retrieved_chunks": list(chunks.values()),
    }
