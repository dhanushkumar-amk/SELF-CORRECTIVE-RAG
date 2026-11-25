"""Credential-free keyword search over indexed PDF passages."""

import re
from typing import Any

from app.models.schemas import ClaimWithSource, RetrievalResult
from app.retrieval.bm25_index import get_bm25_index, tokenize_text

_COMMON_WORDS = set(
    "a an and are about as at be by can could did do does for from had has have how i in "
    "is it me of on or please should the their them this to was were what when where "
    "which who why will with would you".split()
)


def find_passages(query: str, document_id: str | None = None) -> dict[str, Any]:
    terms = set(tokenize_text(query)) - _COMMON_WORDS
    index = get_bm25_index()
    candidates = []
    if terms:
        for match in index.search(" ".join(sorted(terms)), top_k=index.chunk_count):
            metadata = match["metadata"]
            if document_id and metadata["document_id"] != document_id:
                continue
            source = metadata["source_text"]
            for passage in re.split(r"(?<=[.!?])\s+|\n{2,}", source):
                passage = passage.strip()
                overlap = len(terms.intersection(tokenize_text(passage)))
                if overlap and passage:
                    candidates.append((overlap, match["score"], match["id"], passage, metadata))

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
        if len(claims) == 3:
            break

    return {
        "final_status": "source_passages" if claims else "unverifiable",
        "final_answer_text": "\n\n".join(claim.claim_text for claim in claims)
        or "No source passages matched this question. Try a more specific keyword from your documents.",
        "claims": claims,
        "retrieved_chunks": list(chunks.values()),
    }
