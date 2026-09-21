"use client";

import { ShieldCheck, ExternalLink, Heart } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-white/[0.08] bg-black py-12">
      <div className="mx-auto max-w-5xl px-4 sm:px-6">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-white/[0.05] border border-white/10 text-white">
              <ShieldCheck className="h-4 w-4 text-[#0a84ff]" />
            </div>
            <span className="text-xs font-semibold text-white">Self-Correcting RAG</span>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2 text-[10px] font-mono text-zinc-500">
            <span className="rounded-full bg-white/[0.04] px-2.5 py-1 border border-white/[0.06]">Next.js 14</span>
            <span className="rounded-full bg-white/[0.04] px-2.5 py-1 border border-white/[0.06]">DeBERTa-v3 NLI</span>
            <span className="rounded-full bg-white/[0.04] px-2.5 py-1 border border-white/[0.06]">LangGraph</span>
            <span className="rounded-full bg-white/[0.04] px-2.5 py-1 border border-white/[0.06]">FastAPI SSE</span>
            <span className="rounded-full bg-white/[0.04] px-2.5 py-1 border border-white/[0.06]">FAISS + BM25</span>
          </div>

          <div className="flex items-center gap-3 text-xs text-zinc-500">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 hover:text-white transition-colors font-mono"
            >
              Docs <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </div>

        <div className="mt-8 border-t border-white/[0.06] pt-6 text-center text-[11px] text-zinc-600 font-sans">
          © {new Date().getFullYear()} Self-Correcting RAG Engine. Factual Integrity & Hallucination Mitigation.
        </div>
      </div>
    </footer>
  );
}
