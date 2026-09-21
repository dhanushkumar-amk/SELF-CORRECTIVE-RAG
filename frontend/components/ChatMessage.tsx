"use client";

import {
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  FileText,
  ExternalLink,
  User,
  Bot,
  Check,
  X,
  HelpCircle,
  FileCheck,
} from "lucide-react";
import { Button } from "@/components/ui/button";

export interface ClaimItem {
  claim_id?: string;
  claim_text: string;
  chunk_id?: string;
  status: "VERIFIED" | "ENTAILMENT" | "CONTRADICTED" | "NEUTRAL" | "UNVERIFIABLE" | "FAILED";
  confidence_score?: number;
  nli_scores?: {
    entailment: number;
    contradiction: number;
    neutral: number;
  };
  source_text?: string;
  source_page?: number;
  correction_applied?: boolean;
}

export interface ChatMessageData {
  id: string;
  role: "user" | "assistant";
  content: string;
  claims?: ClaimItem[];
  overallStatus?: "fully_verified" | "partially_verified" | "unverifiable" | "uncorrected";
  correctionTriggered?: boolean;
  correctionReason?: string;
  executionTimeMs?: number;
  isStreaming?: boolean;
}

interface ChatMessageProps {
  message: ChatMessageData;
  onInspectChunk?: (claim: ClaimItem) => void;
}

export function ChatMessage({ message, onInspectChunk }: ChatMessageProps) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex items-start justify-end gap-3 max-w-3xl ml-auto">
        <div className="rounded-2xl rounded-tr-sm bg-primary text-primary-foreground px-4 py-3 text-sm shadow-sm font-sans max-w-xl">
          {message.content}
        </div>
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-muted border border-border text-muted-foreground shrink-0">
          <User className="h-4 w-4" />
        </div>
      </div>
    );
  }

  const isVerified = message.overallStatus === "fully_verified";
  const isPartial = message.overallStatus === "partially_verified";

  return (
    <div className="flex items-start gap-3 max-w-3xl mr-auto w-full">
      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground shrink-0 mt-1 shadow-sm">
        <Bot className="h-4 w-4" />
      </div>

      <div className="flex-1 space-y-3 font-sans min-w-0">
        {/* Assistant Header Badge */}
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border pb-2">
          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="font-bold text-foreground">Self-Correcting RAG</span>
            {message.executionTimeMs && (
              <span className="text-muted-foreground text-[11px]">• {message.executionTimeMs}ms</span>
            )}
          </div>

          {message.overallStatus && (
            <span
              className={`inline-flex items-center gap-1 rounded-md px-2.5 py-0.5 text-xs font-mono border ${
                isVerified
                  ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/30"
                  : isPartial
                  ? "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30"
                  : "bg-destructive/10 text-destructive border-destructive/30"
              }`}
            >
              {isVerified ? (
                <CheckCircle2 className="h-3 w-3" />
              ) : (
                <AlertTriangle className="h-3 w-3" />
              )}
              {message.overallStatus.toUpperCase().replace("_", " ")}
            </span>
          )}
        </div>

        {/* Correction Warning Banner */}
        {message.correctionTriggered && (
          <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 flex items-start gap-2.5 text-xs font-mono text-amber-700 dark:text-amber-300">
            <RefreshCw className="h-3.5 w-3.5 text-amber-500 animate-spin mt-0.5 shrink-0" />
            <div>
              <span className="font-bold">DeBERTa NLI Correction Triggered:</span>{" "}
              <span className="font-sans text-muted-foreground">
                {message.correctionReason || "Unverified claims detected. Targeted re-retrieval auto-applied."}
              </span>
            </div>
          </div>
        )}

        {/* Answer Content Paragraph */}
        <div className="rounded-xl border border-border bg-card p-4 text-sm text-foreground leading-relaxed shadow-sm">
          {message.content}
        </div>

        {/* Atomic Claims Inspection */}
        {message.claims && message.claims.length > 0 && (
          <div className="rounded-xl border border-border bg-muted/30 p-4 space-y-2.5 font-mono text-xs">
            <div className="flex items-center justify-between border-b border-border/60 pb-2">
              <span className="font-bold text-foreground flex items-center gap-1.5 uppercase text-[11px] tracking-wider">
                <FileCheck className="h-3.5 w-3.5 text-primary" /> Atomic Claim NLI Breakdown ({message.claims.length})
              </span>
              <span className="text-[10px] text-muted-foreground font-sans">Click chunk to view passage</span>
            </div>

            <div className="space-y-2">
              {message.claims.map((claim, idx) => {
                const claimEnt = claim.status === "VERIFIED" || claim.status === "ENTAILMENT";
                const claimCon = claim.status === "CONTRADICTED";
                const score = claim.confidence_score ?? claim.nli_scores?.entailment;

                return (
                  <div
                    key={idx}
                    className={`rounded-lg border p-3 space-y-1.5 ${
                      claimEnt
                        ? "border-emerald-500/30 bg-emerald-500/5"
                        : claimCon
                        ? "border-destructive/30 bg-destructive/5"
                        : "border-amber-500/30 bg-amber-500/5"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-bold text-muted-foreground">#{idx + 1}</span>
                        <span
                          className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-bold border ${
                            claimEnt
                              ? "bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border-emerald-500/30"
                              : claimCon
                              ? "bg-destructive/20 text-destructive border-destructive/30"
                              : "bg-amber-500/20 text-amber-600 dark:text-amber-400 border-amber-500/30"
                          }`}
                        >
                          {claimEnt ? (
                            <Check className="h-3 w-3" />
                          ) : claimCon ? (
                            <X className="h-3 w-3" />
                          ) : (
                            <HelpCircle className="h-3 w-3" />
                          )}
                          {claim.status}
                        </span>

                        {score !== undefined && (
                          <span className="text-[10px] text-primary bg-primary/10 px-2 py-0.5 rounded border border-primary/20 font-bold">
                            Confidence: {(score * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>

                      {claim.chunk_id && onInspectChunk && (
                        <Button
                          variant="outline"
                          size="xs"
                          onClick={() => onInspectChunk(claim)}
                          className="text-[10px] font-mono border-border text-primary"
                        >
                          <FileText className="mr-1 h-3 w-3" />
                          Chunk: {claim.chunk_id}
                          <ExternalLink className="ml-1 h-2.5 w-2.5" />
                        </Button>
                      )}
                    </div>

                    <p className="text-xs text-foreground bg-background p-2 rounded border border-border">
                      &quot;{claim.claim_text}&quot;
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
