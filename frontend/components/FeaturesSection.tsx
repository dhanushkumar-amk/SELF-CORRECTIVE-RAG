"use client";

import { Database, Split, ShieldCheck, RefreshCw, GitMerge, Cpu, Layers } from "lucide-react";

const FEATURES = [
  {
    icon: Database,
    title: "Hybrid Search & Reranking",
    description:
      "Dense FAISS vector search combined with sparse BM25 keyword matching, reranked by Cross-Encoder for high recall.",
    tag: "Retrieval",
  },
  {
    icon: Split,
    title: "Atomic Claim Decomposition",
    description:
      "Decomposes LLM generated candidate answers into discrete atomic claims and maps each claim to source passage chunks.",
    tag: "Claim Processing",
  },
  {
    icon: ShieldCheck,
    title: "DeBERTa NLI Verification",
    description:
      "Scores premise-hypothesis pairs using local NLI cross-encoders to classify entailment, contradiction, and neutral claims.",
    tag: "NLI Engine",
  },
  {
    icon: RefreshCw,
    title: "LangGraph Correction Loop",
    description:
      "State router triggers targeted claim queries and partial regeneration to eliminate hallucinated facts before output.",
    tag: "Self-Correction",
  },
];

const TECHNICAL_STEPS = [
  {
    step: "01",
    title: "PDF Ingestion",
    desc: "Documents parsed with PyMuPDF, chunked with structural overlap, and indexed into FAISS.",
  },
  {
    step: "02",
    title: "Citation Extraction",
    desc: "LLM synthesizes response structured into discrete claims tagged with source chunk IDs.",
  },
  {
    step: "03",
    title: "NLI Pair Scoring",
    desc: "DeBERTa evaluates premise-hypothesis pairs for exact factual entailment.",
  },
  {
    step: "04",
    title: "State Correction Loop",
    desc: "Failed claims trigger targeted re-retrieval and real-time SSE stream output.",
  },
];

export function FeaturesSection() {
  return (
    <section id="features" className="py-16 border-b border-border">
      <div className="mx-auto max-w-5xl px-4 sm:px-6">
        {/* Header */}
        <div className="text-center max-w-2xl mx-auto">
          <span className="text-[11px] font-mono uppercase tracking-widest text-primary font-bold">
            Subsystem Architecture
          </span>
          <h2 className="mt-2 text-3xl font-bold tracking-tight text-foreground font-mono">
            Engineered for Verifiable Accuracy
          </h2>
          <p className="mt-3 text-sm text-muted-foreground font-sans">
            Four core subsystems operating synchronously inside a compiled LangGraph state machine.
          </p>
        </div>

        {/* 4 Technical Cards */}
        <div className="mt-10 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {FEATURES.map((feat) => {
            const Icon = feat.icon;
            return (
              <div
                key={feat.title}
                className="group relative rounded-xl border border-border bg-card p-5 transition-all hover:border-primary/50 shadow-sm"
              >
                <div className="flex h-9 w-9 items-center justify-center rounded-md bg-muted text-primary mb-3 border border-border">
                  <Icon className="h-4.5 w-4.5" />
                </div>
                <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider">
                  {feat.tag}
                </span>
                <h3 className="mt-1 text-sm font-bold text-foreground font-mono group-hover:text-primary transition-colors">
                  {feat.title}
                </h3>
                <p className="mt-2 text-xs text-muted-foreground leading-relaxed font-sans">
                  {feat.description}
                </p>
              </div>
            );
          })}
        </div>

        {/* Sequence Flow Bar */}
        <div className="mt-10 rounded-xl border border-border bg-card p-6 shadow-sm">
          <h3 className="text-xs font-mono font-bold text-foreground uppercase tracking-wider flex items-center gap-2 mb-4">
            <GitMerge className="h-4 w-4 text-primary" /> State Machine Execution Flow
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {TECHNICAL_STEPS.map((s) => (
              <div key={s.step} className="relative pl-3 border-l-2 border-primary">
                <span className="text-[11px] font-mono font-bold text-primary">{s.step}</span>
                <h4 className="text-xs font-bold text-foreground font-mono mt-0.5">{s.title}</h4>
                <p className="text-[11px] text-muted-foreground font-sans mt-1 leading-normal">{s.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
