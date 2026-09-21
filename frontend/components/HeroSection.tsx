"use client";

import { ArrowRight, ShieldCheck, AlertCircle, CheckCircle2, RefreshCw, Zap, Terminal, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

interface HeroSectionProps {
  onLaunchSandbox: () => void;
}

export function HeroSection({ onLaunchSandbox }: HeroSectionProps) {
  return (
    <section id="overview" className="relative overflow-hidden pt-12 pb-16 md:pt-16 md:pb-24 border-b border-border">
      <div className="mx-auto max-w-5xl px-4 sm:px-6">
        <div className="flex flex-col items-center text-center">
          {/* MCP Mono Pill Tag */}
          <div className="inline-flex items-center gap-2 rounded-md border border-border bg-muted/60 px-3 py-1 text-xs font-mono text-muted-foreground">
            <Sparkles className="h-3.5 w-3.5 text-primary" />
            <span>Self-Correcting RAG Specification v0.1</span>
            <span className="rounded bg-background px-1.5 py-0.5 text-[10px] font-semibold text-foreground border border-border">
              DeBERTa-v3
            </span>
          </div>

          {/* Clean MCP Headline */}
          <h1 className="mt-6 text-4xl font-extrabold tracking-tight text-foreground sm:text-6xl font-mono max-w-4xl leading-tight">
            Factual Grounding for RAG. <br />
            <span className="text-muted-foreground font-sans">Verified by DeBERTa NLI.</span>
          </h1>

          {/* Subtitle */}
          <p className="mt-5 max-w-2xl text-base sm:text-lg text-muted-foreground leading-relaxed font-sans">
            Decompose LLM generated responses into atomic claims, score entailment against source documents using cross-encoder NLI, and automatically trigger targeted re-retrieval & rewriting.
          </p>

          {/* Action Buttons using shadcn Button */}
          <div className="mt-8 flex flex-col sm:flex-row items-center gap-3">
            <Button
              variant="default"
              size="lg"
              onClick={onLaunchSandbox}
              className="w-full sm:w-auto h-10 px-6 font-mono text-xs"
            >
              Open Interactive Sandbox
              <ArrowRight className="ml-1.5 h-3.5 w-3.5" />
            </Button>
            <Button
              variant="outline"
              size="lg"
              onClick={() => {
                document.getElementById("features")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="w-full sm:w-auto h-10 px-6 font-mono text-xs border-border"
            >
              Architecture Spec
            </Button>
          </div>

          {/* Metrics Highlight Bar */}
          <div className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-4 w-full border-t border-border pt-6 text-left font-mono">
            <div className="border-r border-border/40 pr-2">
              <div className="text-xl font-bold text-foreground">100%</div>
              <div className="text-xs text-muted-foreground mt-0.5">Atomic Claim Extraction</div>
            </div>
            <div className="border-r border-border/40 pr-2">
              <div className="text-xl font-bold text-primary">DeBERTa-v3</div>
              <div className="text-xs text-muted-foreground mt-0.5">Cross-Encoder NLI</div>
            </div>
            <div className="border-r border-border/40 pr-2">
              <div className="text-xl font-bold text-foreground">LangGraph</div>
              <div className="text-xs text-muted-foreground mt-0.5">Correction Loop</div>
            </div>
            <div>
              <div className="text-xl font-bold text-emerald-600 dark:text-emerald-400">SSE Stream</div>
              <div className="text-xs text-muted-foreground mt-0.5">Real-time Pipeline</div>
            </div>
          </div>
        </div>

        {/* MCP 2-Color Comparison Box */}
        <div className="mt-12 rounded-xl border border-border bg-card p-6 shadow-sm">
          <div className="flex items-center justify-between border-b border-border pb-3 mb-4">
            <div className="flex items-center gap-2">
              <Terminal className="h-4 w-4 text-primary" />
              <h3 className="text-xs font-mono font-bold text-foreground uppercase tracking-wider">
                Standard RAG vs Self-Correcting RAG
              </h3>
            </div>
            <span className="text-[11px] font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded border border-border">
              Engine Comparison
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-xs">
            {/* Standard RAG Box */}
            <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-destructive flex items-center gap-1.5">
                  <AlertCircle className="h-3.5 w-3.5" /> Standard RAG
                </span>
                <span className="text-[10px] text-destructive bg-destructive/10 px-2 py-0.5 rounded">
                  Unverified
                </span>
              </div>
              <p className="text-foreground font-sans">
                AlphaCorp Q3 revenue was <span className="text-destructive bg-destructive/20 px-1 rounded">$42.5 million</span> with 18% YoY growth.
              </p>
              <div className="pt-2 border-t border-border/40 text-[11px] text-destructive">
                ❌ Source text only contained Q2 figures ($38M)
              </div>
            </div>

            {/* Self-Correcting RAG Box */}
            <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-4 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5">
                  <ShieldCheck className="h-3.5 w-3.5" /> Self-Correcting RAG
                </span>
                <span className="text-[10px] text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">
                  DeBERTa Verified
                </span>
              </div>
              <p className="text-foreground font-sans">
                AlphaCorp Q3 revenue was <span className="text-emerald-600 dark:text-emerald-400 bg-emerald-500/20 px-1 rounded">$45.2 million</span> as verified in Q3 release.
              </p>
              <div className="pt-2 border-t border-border/40 flex items-center justify-between text-[11px] text-emerald-600 dark:text-emerald-400">
                <span className="flex items-center gap-1">
                  <CheckCircle2 className="h-3 w-3" /> Entailed (0.96)
                </span>
                <span className="text-muted-foreground flex items-center gap-1">
                  <RefreshCw className="h-3 w-3 text-primary" /> Auto-Corrected
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
