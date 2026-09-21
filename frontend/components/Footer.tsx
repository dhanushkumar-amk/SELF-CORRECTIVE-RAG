"use client";

import { ShieldCheck, ExternalLink } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-border bg-background py-10 font-mono text-xs">
      <div className="mx-auto max-w-5xl px-4 sm:px-6">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded bg-primary text-primary-foreground">
              <ShieldCheck className="h-3.5 w-3.5" />
            </div>
            <span className="font-bold text-foreground">Self-Correcting RAG</span>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2 text-[10px] text-muted-foreground">
            <span className="rounded bg-muted px-2 py-0.5 border border-border">Next.js 14</span>
            <span className="rounded bg-muted px-2 py-0.5 border border-border">DeBERTa-v3 NLI</span>
            <span className="rounded bg-muted px-2 py-0.5 border border-border">LangGraph</span>
            <span className="rounded bg-muted px-2 py-0.5 border border-border">FastAPI SSE</span>
            <span className="rounded bg-muted px-2 py-0.5 border border-border">FAISS + BM25</span>
          </div>

          <div className="flex items-center gap-3 text-muted-foreground">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 hover:text-foreground transition-colors"
            >
              OpenAPI Docs <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </div>

        <div className="mt-6 border-t border-border pt-4 text-center text-[11px] text-muted-foreground font-sans">
          © {new Date().getFullYear()} Self-Correcting RAG Specification. Model Context Protocol inspired Clean UI.
        </div>
      </div>
    </footer>
  );
}
