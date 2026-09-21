"use client";

import { Database, Split, ShieldCheck, RefreshCw, GitMerge } from "lucide-react";

const FEATURES = [
  {
    icon: Database,
    title: "Hybrid Search & Reranking",
    description:
      "Dense vector search combined with sparse BM25 keyword matching, reranked by Cross-Encoder models.",
    tag: "Retrieval",
    iconColor: "text-[#0a84ff]",
    borderColor: "border-[#0a84ff]/30",
  },
  {
    icon: Split,
    title: "Atomic Claim Extraction",
    description:
      "Extracts standalone factual claims from draft generated answers and maps them to candidate source passage chunks.",
    tag: "Extraction",
    iconColor: "text-[#bf5af2]",
    borderColor: "border-[#bf5af2]/30",
  },
  {
    icon: ShieldCheck,
    title: "DeBERTa NLI Verification",
    description:
      "Evaluates premise-hypothesis pairs using local NLI models to classify entailment, contradiction, and neutral claims.",
    tag: "Verification",
    iconColor: "text-[#30d158]",
    borderColor: "border-[#30d158]/30",
  },
  {
    icon: RefreshCw,
    title: "LangGraph State Correction",
    description:
      "State-machine router triggers targeted claim queries and partial answer regeneration to fix unverified statements.",
    tag: "Self-Correction",
    iconColor: "text-[#ffd60a]",
    borderColor: "border-[#ffd60a]/30",
  },
];

const TECHNICAL_STEPS = [
  {
    step: "01",
    title: "Document Ingestion",
    desc: "PDF files chunked with structural overlap and stored in FAISS + BM25 indices.",
  },
  {
    step: "02",
    title: "Citation Extraction",
    desc: "LLM synthesizes response structured into discrete claims tied to chunk IDs.",
  },
  {
    step: "03",
    title: "NLI Cross-Encoder Scoring",
    desc: "DeBERTa scores premise-hypothesis pairs for exact factual entailment.",
  },
  {
    step: "04",
    title: "Self-Correction Loop",
    desc: "Unverified claims are re-queried, rewritten, and streamed in real-time.",
  },
];

export function FeaturesSection() {
  return (
    <section id="features" className="py-20 border-t border-white/[0.08] relative">
      <div className="mx-auto max-w-5xl px-4 sm:px-6">
        {/* Header */}
        <div className="text-center max-w-2xl mx-auto">
          <span className="text-[11px] font-mono uppercase tracking-widest text-[#0a84ff]">
            Architecture Subsystems
          </span>
          <h2 className="mt-3 text-3xl font-bold tracking-tight text-white sm:text-4xl font-sans">
            Engineered for Precision & Grounding
          </h2>
          <p className="mt-3 text-sm text-zinc-400">
            Four core components operating synchronously inside a compiled LangGraph state machine.
          </p>
        </div>

        {/* 4 Feature Cards */}
        <div className="mt-12 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {FEATURES.map((feat) => {
            const Icon = feat.icon;
            return (
              <div
                key={feat.title}
                className="group relative rounded-2xl border border-white/[0.08] bg-white/[0.02] p-5 backdrop-blur-2xl transition-all duration-300 hover:border-white/20 hover:bg-white/[0.04]"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/[0.05] border border-white/10 mb-4">
                  <Icon className={`h-5 w-5 ${feat.iconColor}`} />
                </div>
                <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider">
                  {feat.tag}
                </span>
                <h3 className="mt-1 text-base font-semibold text-white group-hover:text-[#0a84ff] transition-colors">
                  {feat.title}
                </h3>
                <p className="mt-2 text-xs text-zinc-400 leading-relaxed">
                  {feat.description}
                </p>
              </div>
            );
          })}
        </div>

        {/* Execution Sequence Bar */}
        <div className="mt-12 rounded-3xl border border-white/[0.08] bg-white/[0.02] p-6 sm:p-8 backdrop-blur-2xl">
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2 mb-6">
            <GitMerge className="h-4 w-4 text-[#bf5af2]" /> State Machine Execution Flow
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {TECHNICAL_STEPS.map((s) => (
              <div key={s.step} className="relative pl-4 border-l border-[#0a84ff]/40">
                <span className="text-[11px] font-mono font-bold text-[#0a84ff]">{s.step}</span>
                <h4 className="text-xs font-semibold text-white mt-1">{s.title}</h4>
                <p className="text-[11px] text-zinc-400 mt-1 leading-normal">{s.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
