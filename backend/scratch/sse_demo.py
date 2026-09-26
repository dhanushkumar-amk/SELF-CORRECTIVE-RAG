"""Live SSE demo helper: run a query against the local backend and dump events."""

import json
import sys

import httpx

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:8000"
DOC_ID = "02ef7d1f-11c5-4f82-8583-7245467d84ae"


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "What is the SELF-RAG framework?"
    doc_id = sys.argv[2] if len(sys.argv) > 2 else DOC_ID
    payload = {
        "query": query,
        "document_id": doc_id or None,
        "enable_correction": True,
        "stream": True,
    }
    print(f"== Query: {query!r} (doc={doc_id}) ==")
    with httpx.stream(
        "POST",
        f"{BASE}/api/v1/query",
        json=payload,
        headers={"Accept": "text/event-stream"},
        timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
    ) as resp:
        print(f"HTTP {resp.status_code}")
        if resp.status_code != 200:
            print(resp.text)
            return
        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                print(f"\n[{line[6:].strip()}]")
            elif line.startswith("data:"):
                try:
                    data = json.loads(line[5:])
                except json.JSONDecodeError:
                    print("  (non-JSON data)", line[5:][:200])
                    continue
                if data.get("stage") == "finalize":
                    print(f"  final_status={data['final_status']}")
                    print(f"  answer={data['final_answer_text'][:200]}...")
                    claims = data.get("claims", [])
                    for c in claims:
                        print(
                            f"  claim: [{c.get('verification_status')}] "
                            f"conf={c.get('confidence')} corrected={c.get('was_corrected')} :: {c['claim_text'][:80]}"
                        )
                    print(f"  latency_ms={data.get('latency_ms')}")
                else:
                    slim = {
                        k: v
                        for k, v in data.items()
                        if k not in ("chunks", "claims", "failed_claims", "retrieved_chunks")
                    }
                    print(f"  {json.dumps(slim)[:160]}")


if __name__ == "__main__":
    main()
