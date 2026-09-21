"use client";

import { ArrowRight, ShieldCheck, AlertCircle, CheckCircle2, RefreshCw, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";

interface HeroSectionProps {
  onLaunchSandbox: () => void;
}

export function HeroSection({ onLaunchSandbox }: HeroSectionProps) {
  return (
    <section id="overview" className="relative overflow-hidden pt-16 pb-20 md:pt-24 md:pb-28">
      {/* Subtle Apple Ambient Lighting */}
      <div className="absolute top-1/3 left-1/2 -z-10 h-[300px] w-[500px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[#0a84ff]/10 blur-[140px] pointer-events-none" />

      <div className="mx-auto max-w-5xl px-4 sm:px-6">
        <div className="flex flex-col items-center text-center">
          {/* Apple Style Top Pill Badge */}
          <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3.5 py-1 text-xs text-zinc-300 backdrop-blur-xl">
            <span className="flex h-2 w-2 rounded-full bg-[#0a84ff] shadow-[0_0_8px_#0a84ff]" />
            <span className="font-mono text-[11px] text-zinc-400">DeBERTa-v3 Natural Language Inference</span>
          </div>

          {/* Headline */}
          <h1 className="mt-8 text-4xl font-extrabold tracking-tight text-white sm:text-6xl lg:text-7xl font-sans max-w-4xl leading-[1.08]">
            Verifiable AI Grounding.{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#0a84ff] via-[#5e5ce6] to-[#bf5af2]">
              Zero Hallucinations.
            </span>
          </h1>

          {/* Subtitle */}
          <p className="mt-6 max-w-2xl text-base sm:text-lg text-zinc-400 leading-relaxed font-normal">
            Decompose model outputs into discrete claims, verify factual entailment against source documents using DeBERTa Cross-Encoder NLI, and automatically trigger targeted re-retrieval & rewriting.
          </p>

          {/* Action Buttons */}
          <div className="mt-8 flex flex-col sm:flex-row items-center gap-3">
            <Button
              size="lg"
              onClick={onLaunchSandbox}
              className="w-full sm:w-auto h-11 px-7 bg-white hover:bg-zinc-200 text-black font-semibold text-xs rounded-full shadow-lg shadow-white/5 active:scale-[0.98] transition-all"
            >
              Open Interactive Sandbox
              <ArrowRight className="ml-1.5 h-3.5 w-3.5" />
            </Button>
            <a
              href="#features"
              className="w-full sm:w-auto inline-flex items-center justify-center h-11 px-6 rounded-full border border-white/10 bg-white/[0.04] text-zinc-300 text-xs font-medium hover:bg-white/[0.08] hover:text-white transition-all"
            >
              System Architecture
            </a>
          </div>

          {/* Metrics Highlight Bar */}
          <div className="mt-14 grid grid-cols-2 md:grid-cols-4 gap-6 w-full border-t border-b border-white/[0.08] py-6 text-left">
            <div>
              <div className="text-2xl font-semibold text-white font-mono">100%</div>
              <div className="text-xs text-zinc-500 mt-0.5 font-sans">Atomic Claim Extraction</div>
            </div>
            <div>
              <div className="text-2xl font-semibold text-[#0a84ff] font-mono">DeBERTa-v3</div>
              <div className="text-xs text-zinc-500 mt-0.5 font-sans">Cross-Encoder NLI</div>
            </div>
            <div>
              <div className="text-2xl font-semibold text-[#bf5af2] font-mono">LangGraph</div>
              <div className="text-xs text-zinc-500 mt-0.5 font-sans">Self-Correction State Machine</div>
            </div>
            <div>
              <div className="text-2xl font-semibold text-[#30d158] font-mono">SSE Stream</div>
              <div className="text-xs text-zinc-500 mt-0.5 font-sans">Real-time Step Pipeline</div>
            </div>
          </div>
        </div>

        {/* Apple Style Visual Comparison Card */}
        <div className="mt-14 rounded-3xl border border-white/[0.08] bg-white/[0.02] p-6 sm:p-8 backdrop-blur-2xl">
          <div className="flex items-center justify-between mb-6 border-b border-white/[0.06] pb-4">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-[#ffd60a]" />
              <h3 className="text-sm font-semibold text-white">Pipeline Factual Guardrail Comparison</h3>
            </div>
            <span className="text-[11px] font-mono text-zinc-500 bg-white/[0.04] px-2.5 py-1 rounded-full border border-white/[0.06]">
              Live Engine Test
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Standard RAG Box */}
            <div className="rounded-2xl border border-[#ff453a]/20 bg-[#ff453a]/[0.03] p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold text-[#ff453a] flex items-center gap-1.5">
                  <AlertCircle className="h-4 w-4" /> Standard RAG
                </span>
                <span className="text-[10px] font-mono text-[#ff453a] bg-[#ff453a]/10 px-2 py-0.5 rounded-full border border-[#ff453a]/20">
                  Hallucination Risk
                </span>
              </div>
              <p className="text-xs text-zinc-400 mb-3">
                Prompt: <span className="text-zinc-200 italic">&quot;What were Q3 revenues for AlphaCorp?&quot;</span>
              </p>
              <div className="text-xs font-mono bg-black/60 p-3.5 rounded-xl border border-white/[0.06] space-y-2">
                <p className="text-zinc-300">
                  AlphaCorp reported Q3 revenue of <span className="text-[#ff453a] bg-[#ff453a]/20 px-1 py-0.5 rounded">$42.5 million</span> with 18% YoY growth.
                </p>
                <div className="pt-2 border-t border-white/[0.06] text-[11px] text-[#ff453a]/80">
                  ❌ Source document only contained Q2 figures ($38M)
                </div>
              </div>
            </div>

            {/* Self-Correcting RAG Box */}
            <div className="rounded-2xl border border-[#30d158]/30 bg-[#30d158]/[0.03] p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold text-[#30d158] flex items-center gap-1.5">
                  <ShieldCheck className="h-4 w-4 text-[#30d158]" /> Self-Correcting RAG (DeBERTa)
                </span>
                <span className="text-[10px] font-mono text-[#30d158] bg-[#30d158]/10 px-2 py-0.5 rounded-full border border-[#30d158]/20">
                  Auto-Corrected
                </span>
              </div>
              <p className="text-xs text-zinc-400 mb-3">
                Prompt: <span className="text-zinc-200 italic">&quot;What were Q3 revenues for AlphaCorp?&quot;</span>
              </p>
              <div className="text-xs font-mono bg-black/60 p-3.5 rounded-xl border border-[#30d158]/20 space-y-2">
                <p className="text-zinc-300">
                  AlphaCorp reported Q3 revenue of <span className="text-[#30d158] bg-[#30d158]/20 px-1 py-0.5 rounded">$45.2 million</span> as verified in Q3 release.
                </p>
                <div className="pt-2 border-t border-white/[0.06] flex items-center justify-between text-[11px] text-[#30d158]">
                  <span className="flex items-center gap-1">
                    <CheckCircle2 className="h-3.5 w-3.5" /> Entailed (0.96)
                  </span>
                  <span className="text-zinc-400 flex items-center gap-1">
                    <RefreshCw className="h-3 w-3 text-[#0a84ff]" /> Targeted Search
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
