"use client";

import { useState, useRef } from "react";
import {
  Send,
  Sparkles,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  FileText,
  FileCheck,
  Search,
  Check,
  HelpCircle,
  X,
  ExternalLink,
  Zap,
  SlidersHorizontal,
  BarChart2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { AnalyticsCharts } from "@/components/AnalyticsCharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface ClaimItem {
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

interface FinalAnswerPayload {
  answer: string;
  claims: ClaimItem[];
  overall_status: "fully_verified" | "partially_verified" | "unverifiable" | "uncorrected";
  total_claims_count?: number;
  verified_claims_count?: number;
  correction_triggered?: boolean;
  correlation_id?: string;
  execution_time_ms?: number;
}

interface QueryConsoleProps {
  selectedDocumentId?: string;
}

const SAMPLE_QUERIES = [
  {
    label: "Earnings & Revenue",
    query: "What was AlphaCorp's Q3 revenue and YoY growth rate according to the earnings report?",
  },
  {
    label: "Contradiction Test",
    query: "Is the server deployment process completely automated with zero human approval?",
  },
  {
    label: "Fact Checking",
    query: "What are the supported maximum payload size limits for API endpoints?",
  },
];

export function QueryConsole({ selectedDocumentId }: QueryConsoleProps) {
  const [query, setQuery] = useState("");
  const [enableCorrection, setEnableCorrection] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [showAnalytics, setShowAnalytics] = useState(true);

  const [activeStep, setActiveStep] = useState<
    "idle" | "retrieving" | "generating" | "verifying" | "correcting" | "completed" | "error"
  >("idle");
  const [stepLogs, setStepLogs] = useState<string[]>([]);
  const [correctionBanner, setCorrectionBanner] = useState<{
    triggered: boolean;
    reason?: string;
  }>({ triggered: false });

  const [finalOutput, setFinalOutput] = useState<FinalAnswerPayload | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [activeChunkModal, setActiveChunkModal] = useState<ClaimItem | null>(null);
  const consoleEndRef = useRef<HTMLDivElement>(null);

  const handleRunQuery = async (queryText?: string) => {
    const promptToUse = queryText || query;
    if (!promptToUse.trim()) return;

    setIsStreaming(true);
    setActiveStep("retrieving");
    setStepLogs(["[INIT] Connection opened to SSE pipeline endpoint..."]);
    setCorrectionBanner({ triggered: false });
    setFinalOutput(null);
    setErrorMsg(null);

    try {
      const response = await fetch(`${API_URL}/api/v1/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          query: promptToUse,
          document_id: selectedDocumentId || null,
          enable_correction: enableCorrection,
          stream: true,
        }),
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || errJson.message || `Query failed (${response.status})`);
      }

      if (!response.body) {
        throw new Error("No readable stream response received from API");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const block of lines) {
          if (!block.trim()) continue;

          let eventName = "message";
          let dataStr = "";

          for (const line of block.split("\n")) {
            if (line.startsWith("event:")) {
              eventName = line.replace("event:", "").trim();
            } else if (line.startsWith("data:")) {
              dataStr += line.replace("data:", "").trim();
            }
          }

          if (!dataStr) continue;

          try {
            const data = JSON.parse(dataStr);
            const currentEventType = data.event || eventName || data.type;

            if (currentEventType === "retrieval_started") {
              setActiveStep("retrieving");
              setStepLogs((prev) => [...prev, `[RETRIEVAL] Hybrid dense FAISS + BM25 search running...`]);
            } else if (currentEventType === "retrieval_complete") {
              setStepLogs((prev) => [
                ...prev,
                `[RETRIEVAL COMPLETE] Retained top ${data.total_chunks || data.chunk_count || 3} reranked passages.`,
              ]);
            } else if (currentEventType === "generation_started") {
              setActiveStep("generating");
              setStepLogs((prev) => [...prev, `[GENERATION] Synthesizing draft answer with claim citations...`]);
            } else if (currentEventType === "generation_complete") {
              setStepLogs((prev) => [
                ...prev,
                `[GENERATION COMPLETE] Extracted ${data.claim_count || data.claims?.length || 0} atomic claims.`,
              ]);
            } else if (currentEventType === "verification_started") {
              setActiveStep("verifying");
              setStepLogs((prev) => [...prev, `[NLI VERIFICATION] Scoring claims via DeBERTa Cross-Encoder...`]);
            } else if (currentEventType === "verification_complete") {
              setStepLogs((prev) => [
                ...prev,
                `[NLI COMPLETE] Verification finished. Fully Verified: ${data.fully_verified ? "YES" : "NO"}`,
              ]);
            } else if (currentEventType === "correction_triggered") {
              setActiveStep("correcting");
              setCorrectionBanner({
                triggered: true,
                reason: data.reason || "Hallucinated or unverified claims detected. Running re-retrieval loop...",
              });
              setStepLogs((prev) => [
                ...prev,
                `[CORRECTION LOOP] Unverified claims detected! Triggering targeted re-retrieval...`,
              ]);
            } else if (currentEventType === "final_answer" || data.answer) {
              setActiveStep("completed");
              setStepLogs((prev) => [...prev, `[COMPLETE] Final verified response payload ready.`]);
              setFinalOutput({
                answer: data.answer || data.content || "",
                claims: data.claims || [],
                overall_status: data.overall_status || (data.fully_verified ? "fully_verified" : "partially_verified"),
                total_claims_count: data.total_claims_count || data.claims?.length,
                verified_claims_count: data.verified_claims_count,
                correction_triggered: data.correction_triggered || false,
                correlation_id: data.correlation_id,
                execution_time_ms: data.execution_time_ms,
              });
            } else if (currentEventType === "error") {
              setActiveStep("error");
              setErrorMsg(data.message || data.detail || "Pipeline processing error");
            }
          } catch {
            // Skip invalid JSON
          }
        }
      }
    } catch (err: any) {
      setActiveStep("error");
      setErrorMsg(err?.message || "Error communicating with server");
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <section id="sandbox" className="py-14 border-b border-border">
      <div className="mx-auto max-w-5xl px-4 sm:px-6 space-y-6">
        {/* Header & Controls */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 font-mono text-xs">
              <span className="rounded bg-muted px-2 py-0.5 text-muted-foreground border border-border">
                Interactive Playground
              </span>
              {selectedDocumentId && (
                <span className="text-primary bg-primary/10 px-2 py-0.5 rounded border border-primary/20">
                  Target: {selectedDocumentId.slice(0, 8)}...
                </span>
              )}
            </div>
            <h2 className="text-2xl font-bold tracking-tight text-foreground font-mono mt-1 flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" /> Self-Correcting RAG Console
            </h2>
          </div>

          <div className="flex items-center gap-3 font-mono text-xs">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowAnalytics(!showAnalytics)}
              className="border-border text-xs"
            >
              <BarChart2 className="mr-1.5 h-3.5 w-3.5 text-primary" />
              {showAnalytics ? "Hide Charts" : "Show Analytics"}
            </Button>

            <div className="flex items-center gap-2 bg-muted p-1.5 rounded-lg border border-border">
              <SlidersHorizontal className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-foreground font-medium">NLI Self-Correction Loop</span>
              <button
                type="button"
                onClick={() => setEnableCorrection(!enableCorrection)}
                className={`relative inline-flex h-4 w-7 shrink-0 cursor-pointer rounded-full border border-transparent transition-colors duration-200 ease-in-out ${
                  enableCorrection ? "bg-primary" : "bg-muted-foreground/30"
                }`}
              >
                <span
                  className={`pointer-events-none inline-block h-3 w-3 transform rounded-full bg-background shadow transition duration-200 ease-in-out ${
                    enableCorrection ? "translate-x-3" : "translate-x-0"
                  }`}
                />
              </button>
            </div>
          </div>
        </div>

        {/* Sample Prompt Shortcuts */}
        <div className="flex flex-wrap items-center gap-2 font-mono">
          <span className="text-xs text-muted-foreground mr-1 font-sans">Prompt Shortcuts:</span>
          {SAMPLE_QUERIES.map((sample) => (
            <Button
              key={sample.label}
              variant="outline"
              size="xs"
              onClick={() => {
                setQuery(sample.query);
                handleRunQuery(sample.query);
              }}
              disabled={isStreaming}
              className="border-border text-xs text-foreground hover:bg-muted"
            >
              <Search className="mr-1 h-3 w-3 text-primary" />
              {sample.label}
            </Button>
          ))}
        </div>

        {/* Search Console Input Bar */}
        <div className="relative">
          <div className="relative flex items-center rounded-xl border border-border bg-card shadow-sm focus-within:border-primary focus-within:ring-1 focus-within:ring-primary transition-all">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !isStreaming) handleRunQuery();
              }}
              placeholder="Ask document query (e.g. 'What were Q3 revenues and YoY growth rate?')..."
              className="w-full bg-transparent px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none font-sans"
            />
            <Button
              variant="default"
              size="sm"
              onClick={() => handleRunQuery()}
              disabled={isStreaming || !query.trim()}
              className="mr-2 font-mono text-xs"
            >
              {isStreaming ? (
                <>
                  <RefreshCw className="mr-1.5 h-3.5 w-3.5 animate-spin" /> Processing...
                </>
              ) : (
                <>
                  <Send className="mr-1.5 h-3.5 w-3.5" /> Run Query
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Analytics Charts Subsystem */}
        {showAnalytics && (
          <AnalyticsCharts
            claims={finalOutput?.claims}
            executionTimeMs={finalOutput?.execution_time_ms}
            totalClaimsCount={finalOutput?.total_claims_count}
            verifiedCount={finalOutput?.verified_claims_count}
          />
        )}

        {/* Pipeline Step Timeline */}
        {(isStreaming || activeStep !== "idle") && (
          <div className="rounded-xl border border-border bg-card p-5 space-y-3 shadow-sm">
            <div className="flex items-center justify-between border-b border-border pb-2.5 font-mono">
              <h3 className="text-xs font-bold uppercase tracking-wider text-primary flex items-center gap-1.5">
                <Zap className="h-3.5 w-3.5" /> LangGraph SSE Execution Pipeline
              </h3>
              <span className="text-[10px] text-muted-foreground uppercase">
                State: <span className="text-foreground font-bold">{activeStep}</span>
              </span>
            </div>

            {/* Step Pills */}
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 font-mono">
              {[
                { id: "retrieving", label: "Hybrid Search", icon: Search },
                { id: "generating", label: "Generation", icon: FileText },
                { id: "verifying", label: "DeBERTa NLI", icon: ShieldCheck },
                { id: "correcting", label: "Correction Loop", icon: RefreshCw },
                { id: "completed", label: "Verified Output", icon: CheckCircle2 },
              ].map((step) => {
                const Icon = step.icon;
                const isCurrent = activeStep === step.id;
                const isPassed =
                  (activeStep === "generating" && step.id === "retrieving") ||
                  (activeStep === "verifying" && (step.id === "retrieving" || step.id === "generating")) ||
                  (activeStep === "correcting" && step.id !== "completed") ||
                  (activeStep === "completed" && step.id !== "correcting") ||
                  (activeStep === "completed" && correctionBanner.triggered && step.id === "correcting");

                return (
                  <div
                    key={step.id}
                    className={`flex items-center gap-1.5 p-2 rounded-lg border text-[11px] font-medium transition-all ${
                      isCurrent
                        ? "border-primary bg-primary/10 text-primary font-bold animate-pulse"
                        : isPassed
                        ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                        : "border-border bg-muted/30 text-muted-foreground"
                    }`}
                  >
                    <Icon className={`h-3.5 w-3.5 ${isCurrent ? "animate-spin text-primary" : ""}`} />
                    <span className="truncate">{step.label}</span>
                  </div>
                );
              })}
            </div>

            {/* Terminal Streaming Logs */}
            <div className="rounded-lg border border-border bg-muted/60 p-3 font-mono text-[11px] text-foreground space-y-1 max-h-32 overflow-y-auto">
              {stepLogs.map((log, idx) => (
                <div key={idx} className="flex items-start gap-2">
                  <span className="text-primary font-bold select-none">&gt;</span>
                  <span className="break-words">{log}</span>
                </div>
              ))}
              <div ref={consoleEndRef} />
            </div>
          </div>
        )}

        {/* Correction Warning Banner */}
        {correctionBanner.triggered && (
          <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 p-4 flex items-start gap-3 font-mono text-xs text-amber-700 dark:text-amber-300">
            <RefreshCw className="h-4 w-4 text-amber-500 animate-spin mt-0.5 shrink-0" />
            <div>
              <h4 className="font-bold">NLI Self-Correction Loop Triggered</h4>
              <p className="text-[11px] text-muted-foreground mt-0.5 font-sans leading-relaxed">
                {correctionBanner.reason ||
                  "One or more extracted claims failed DeBERTa entailment verification. Triggering targeted re-retrieval and rewriting answer..."}
              </p>
            </div>
          </div>
        )}

        {/* Error Alert */}
        {errorMsg && (
          <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-4 flex items-center justify-between text-xs text-destructive font-mono">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              <span>{errorMsg}</span>
            </div>
            <Button variant="ghost" size="xs" onClick={() => setErrorMsg(null)}>
              Dismiss
            </Button>
          </div>
        )}

        {/* Final Output Synthesis Card */}
        {finalOutput && (
          <div className="space-y-4">
            <div className="rounded-xl border border-border bg-card p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between border-b border-border pb-3">
                <div className="flex items-center gap-2 font-mono">
                  <ShieldCheck className="h-5 w-5 text-primary" />
                  <h3 className="text-sm font-bold text-foreground">Verified Synthesis</h3>
                </div>

                <span
                  className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1 text-xs font-mono border ${
                    finalOutput.overall_status === "fully_verified"
                      ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/30"
                      : finalOutput.overall_status === "partially_verified"
                      ? "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30"
                      : "bg-destructive/10 text-destructive border-destructive/30"
                  }`}
                >
                  {finalOutput.overall_status === "fully_verified" ? (
                    <CheckCircle2 className="h-3.5 w-3.5" />
                  ) : (
                    <AlertTriangle className="h-3.5 w-3.5" />
                  )}
                  {finalOutput.overall_status.toUpperCase().replace("_", " ")}
                </span>
              </div>

              <div className="text-foreground text-sm leading-relaxed font-sans bg-muted/40 p-4 rounded-lg border border-border">
                {finalOutput.answer}
              </div>
            </div>

            {/* Atomic Claim Inspection */}
            <div className="rounded-xl border border-border bg-card p-6 shadow-sm space-y-3 font-mono">
              <div className="flex items-center justify-between border-b border-border pb-2.5">
                <h4 className="text-xs font-bold text-foreground flex items-center gap-2 uppercase tracking-wider">
                  <FileCheck className="h-4 w-4 text-primary" /> Atomic Claim Breakdown & NLI Scores
                </h4>
                <span className="text-[11px] text-muted-foreground font-sans">Click chunk tag to view passage</span>
              </div>

              <div className="space-y-2.5">
                {finalOutput.claims && finalOutput.claims.length > 0 ? (
                  finalOutput.claims.map((c, i) => {
                    const isVerified = c.status === "VERIFIED" || c.status === "ENTAILMENT";
                    const isContradiction = c.status === "CONTRADICTED";

                    return (
                      <div
                        key={i}
                        className={`rounded-lg border p-3.5 space-y-2 ${
                          isVerified
                            ? "border-emerald-500/30 bg-emerald-500/5"
                            : isContradiction
                            ? "border-destructive/30 bg-destructive/5"
                            : "border-amber-500/30 bg-amber-500/5"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-bold text-muted-foreground">Claim #{i + 1}</span>
                            <span
                              className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-bold border ${
                                isVerified
                                  ? "bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border-emerald-500/30"
                                  : isContradiction
                                  ? "bg-destructive/20 text-destructive border-destructive/30"
                                  : "bg-amber-500/20 text-amber-600 dark:text-amber-400 border-amber-500/30"
                              }`}
                            >
                              {isVerified ? (
                                <Check className="h-3 w-3" />
                              ) : isContradiction ? (
                                <X className="h-3 w-3" />
                              ) : (
                                <HelpCircle className="h-3 w-3" />
                              )}
                              {c.status}
                            </span>

                            {c.confidence_score !== undefined && (
                              <span className="text-[10px] text-primary bg-primary/10 px-2 py-0.5 rounded border border-primary/20 font-bold">
                                Entailment: {(c.confidence_score * 100).toFixed(0)}%
                              </span>
                            )}
                          </div>

                          {c.chunk_id && (
                            <Button
                              variant="outline"
                              size="xs"
                              onClick={() => setActiveChunkModal(c)}
                              className="text-[11px] font-mono border-border text-primary"
                            >
                              <FileText className="mr-1 h-3 w-3" />
                              Chunk: {c.chunk_id}
                              <ExternalLink className="ml-1 h-2.5 w-2.5" />
                            </Button>
                          )}
                        </div>

                        <p className="text-xs text-foreground bg-background p-2.5 rounded border border-border">
                          &quot;{c.claim_text}&quot;
                        </p>
                      </div>
                    );
                  })
                ) : (
                  <p className="text-xs text-muted-foreground italic font-sans">No discrete claims extracted.</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Source Chunk Inspection Modal */}
        {activeChunkModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 font-mono">
            <div className="relative w-full max-w-xl rounded-xl border border-border bg-card p-6 shadow-xl space-y-4">
              <div className="flex items-center justify-between border-b border-border pb-3">
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4 text-primary" />
                  <h3 className="text-xs font-bold text-foreground">
                    Source Passage Chunk [{activeChunkModal.chunk_id}]
                  </h3>
                </div>
                <Button variant="ghost" size="icon-xs" onClick={() => setActiveChunkModal(null)}>
                  <X className="h-4 w-4 text-muted-foreground" />
                </Button>
              </div>

              <div className="space-y-1">
                <span className="text-[10px] font-bold text-muted-foreground uppercase">Target Claim:</span>
                <p className="text-xs text-primary bg-primary/10 p-2.5 rounded border border-primary/20">
                  {activeChunkModal.claim_text}
                </p>
              </div>

              <div className="space-y-1">
                <span className="text-[10px] font-bold text-muted-foreground uppercase">Ground-Truth Passage:</span>
                <div className="text-xs text-foreground bg-muted/60 p-3 rounded-lg border border-border max-h-48 overflow-y-auto leading-relaxed">
                  {activeChunkModal.source_text || "Source passage text content registered in database index."}
                </div>
              </div>

              <div className="flex justify-end pt-2">
                <Button variant="outline" size="sm" onClick={() => setActiveChunkModal(null)}>
                  Close
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
