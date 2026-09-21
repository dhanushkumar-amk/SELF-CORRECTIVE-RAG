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
} from "lucide-react";
import { Button } from "@/components/ui/button";

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
    <section id="sandbox" className="py-16 border-t border-white/[0.08] relative">
      <div className="mx-auto max-w-5xl px-4 sm:px-6">
        {/* Header & Controls */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
          <div>
            <div className="flex items-center gap-2">
              <span className="rounded-full bg-white/[0.06] px-2.5 py-0.5 text-[10px] font-mono text-zinc-400 border border-white/[0.08]">
                Interactive Playground
              </span>
              {selectedDocumentId && (
                <span className="text-[10px] font-mono text-[#0a84ff] bg-[#0a84ff]/10 px-2 py-0.5 rounded-full border border-[#0a84ff]/20">
                  Target: {selectedDocumentId.slice(0, 8)}...
                </span>
              )}
            </div>
            <h2 className="text-2xl font-bold tracking-tight text-white mt-1.5 flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-[#0a84ff]" /> Self-Correcting RAG Console
            </h2>
          </div>

          <div className="flex items-center gap-2 bg-white/[0.03] p-2 rounded-full border border-white/[0.08] backdrop-blur-xl">
            <SlidersHorizontal className="h-3.5 w-3.5 text-zinc-400 ml-1" />
            <span className="text-xs text-zinc-300 font-medium">NLI Self-Correction Loop</span>
            <button
              type="button"
              onClick={() => setEnableCorrection(!enableCorrection)}
              className={`relative inline-flex h-4.5 w-8 shrink-0 cursor-pointer rounded-full border border-transparent transition-colors duration-200 ease-in-out ${
                enableCorrection ? "bg-[#0071e3]" : "bg-white/20"
              }`}
            >
              <span
                className={`pointer-events-none inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition duration-200 ease-in-out ${
                  enableCorrection ? "translate-x-3.5" : "translate-x-0"
                }`}
              />
            </button>
          </div>
        </div>

        {/* Sample Prompt Shortcuts */}
        <div className="mb-5 flex flex-wrap items-center gap-2">
          <span className="text-xs text-zinc-500 mr-1">Prompts:</span>
          {SAMPLE_QUERIES.map((sample) => (
            <button
              key={sample.label}
              onClick={() => {
                setQuery(sample.query);
                handleRunQuery(sample.query);
              }}
              disabled={isStreaming}
              className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-xs text-zinc-300 hover:bg-white/[0.08] hover:text-white transition-all"
            >
              <Search className="h-3 w-3 text-[#0a84ff]" />
              <span>{sample.label}</span>
            </button>
          ))}
        </div>

        {/* Apple Spotlight Search Bar */}
        <div className="relative mb-8">
          <div className="relative flex items-center rounded-2xl border border-white/15 bg-white/[0.03] shadow-2xl backdrop-blur-2xl focus-within:border-[#0a84ff] focus-within:ring-2 focus-within:ring-[#0a84ff]/20 transition-all">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !isStreaming) handleRunQuery();
              }}
              placeholder="Ask document question (e.g. 'What were Q3 revenues and YoY growth?')..."
              className="w-full bg-transparent px-5 py-3.5 text-sm text-white placeholder:text-zinc-500 focus:outline-none"
            />
            <Button
              onClick={() => handleRunQuery()}
              disabled={isStreaming || !query.trim()}
              className="mr-2.5 bg-[#0071e3] hover:bg-[#0077ed] text-white font-medium text-xs rounded-xl px-4 h-9 shadow-sm"
            >
              {isStreaming ? (
                <>
                  <RefreshCw className="mr-1.5 h-3.5 w-3.5 animate-spin" /> Running...
                </>
              ) : (
                <>
                  <Send className="mr-1.5 h-3.5 w-3.5" /> Run Query
                </>
              )}
            </Button>
          </div>
        </div>

        {/* SSE Pipeline Timeline */}
        {(isStreaming || activeStep !== "idle") && (
          <div className="mb-8 rounded-3xl border border-white/[0.08] bg-white/[0.02] p-6 backdrop-blur-2xl">
            <div className="flex items-center justify-between border-b border-white/[0.06] pb-3 mb-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[#0a84ff] flex items-center gap-1.5">
                <Zap className="h-3.5 w-3.5" /> LangGraph State Pipeline
              </h3>
              <span className="text-[10px] font-mono text-zinc-500 uppercase">
                State: <span className="text-white font-semibold">{activeStep}</span>
              </span>
            </div>

            {/* Step Pills */}
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 mb-4">
              {[
                { id: "retrieving", label: "Hybrid Search", icon: Search },
                { id: "generating", label: "Generation", icon: FileText },
                { id: "verifying", label: "DeBERTa NLI", icon: ShieldCheck },
                { id: "correcting", label: "Correction Loop", icon: RefreshCw },
                { id: "completed", label: "Verified Response", icon: CheckCircle2 },
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
                    className={`flex items-center gap-2 p-2 rounded-xl border text-[11px] font-medium transition-all ${
                      isCurrent
                        ? "border-[#0a84ff] bg-[#0a84ff]/20 text-white animate-pulse"
                        : isPassed
                        ? "border-[#30d158]/30 bg-[#30d158]/10 text-[#30d158]"
                        : "border-white/[0.06] bg-white/[0.02] text-zinc-500"
                    }`}
                  >
                    <Icon className={`h-3.5 w-3.5 ${isCurrent ? "animate-spin text-[#0a84ff]" : ""}`} />
                    <span className="truncate">{step.label}</span>
                  </div>
                );
              })}
            </div>

            {/* Streaming Log Box */}
            <div className="rounded-xl border border-white/[0.08] bg-black/80 p-3.5 font-mono text-[11px] text-zinc-300 space-y-1 max-h-36 overflow-y-auto">
              {stepLogs.map((log, idx) => (
                <div key={idx} className="flex items-start gap-2">
                  <span className="text-[#0a84ff] select-none">&gt;</span>
                  <span className="break-words">{log}</span>
                </div>
              ))}
              <div ref={consoleEndRef} />
            </div>
          </div>
        )}

        {/* Correction Alert Banner */}
        {correctionBanner.triggered && (
          <div className="mb-8 rounded-2xl border border-[#ffd60a]/30 bg-[#ffd60a]/10 p-4 backdrop-blur-xl flex items-start gap-3">
            <RefreshCw className="h-5 w-5 text-[#ffd60a] animate-spin mt-0.5 shrink-0" />
            <div>
              <h4 className="text-xs font-semibold text-[#ffd60a]">
                NLI Correction Loop Activated
              </h4>
              <p className="text-[11px] text-[#ffd60a]/80 mt-0.5 leading-relaxed">
                {correctionBanner.reason ||
                  "One or more extracted claims failed DeBERTa entailment verification. Re-retrieving candidate context and regenerating answer facts..."}
              </p>
            </div>
          </div>
        )}

        {/* Error Alert */}
        {errorMsg && (
          <div className="mb-8 rounded-2xl border border-[#ff453a]/30 bg-[#ff453a]/10 p-4 flex items-center justify-between text-xs text-[#ff453a]">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-[#ff453a]" />
              <span>{errorMsg}</span>
            </div>
            <button onClick={() => setErrorMsg(null)} className="text-[#ff453a] hover:text-white">
              Dismiss
            </button>
          </div>
        )}

        {/* Output Answer & Atomic Claims */}
        {finalOutput && (
          <div className="space-y-5">
            {/* Verified Answer Card */}
            <div className="rounded-3xl border border-white/[0.08] bg-white/[0.02] p-6 sm:p-8 backdrop-blur-2xl">
              <div className="flex items-center justify-between border-b border-white/[0.06] pb-4 mb-5">
                <div className="flex items-center gap-2.5">
                  <div className="p-1.5 rounded-lg bg-[#0a84ff]/10 text-[#0a84ff] border border-[#0a84ff]/20">
                    <ShieldCheck className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-white">Verified Answer Synthesis</h3>
                    <p className="text-[11px] font-mono text-zinc-400">
                      {finalOutput.claims?.length || 0} Atomic Claims Verified
                    </p>
                  </div>
                </div>

                <span
                  className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold border ${
                    finalOutput.overall_status === "fully_verified"
                      ? "bg-[#30d158]/15 text-[#30d158] border-[#30d158]/30"
                      : finalOutput.overall_status === "partially_verified"
                      ? "bg-[#ffd60a]/15 text-[#ffd60a] border-[#ffd60a]/30"
                      : "bg-[#ff453a]/15 text-[#ff453a] border-[#ff453a]/30"
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

              <div className="text-zinc-200 text-sm leading-relaxed bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                {finalOutput.answer}
              </div>
            </div>

            {/* Atomic Claims Inspection */}
            <div className="rounded-3xl border border-white/[0.08] bg-white/[0.02] p-6 sm:p-8 backdrop-blur-2xl">
              <div className="flex items-center justify-between mb-5">
                <h4 className="text-sm font-bold text-white flex items-center gap-2">
                  <FileCheck className="h-4 w-4 text-[#0a84ff]" /> Atomic Claim DeBERTa NLI Breakdown
                </h4>
                <span className="text-[11px] text-zinc-500">Click chunk tag to view source passage</span>
              </div>

              <div className="space-y-3">
                {finalOutput.claims && finalOutput.claims.length > 0 ? (
                  finalOutput.claims.map((c, i) => {
                    const isVerified = c.status === "VERIFIED" || c.status === "ENTAILMENT";
                    const isContradiction = c.status === "CONTRADICTED";

                    return (
                      <div
                        key={i}
                        className={`rounded-2xl border p-4 transition-all ${
                          isVerified
                            ? "border-[#30d158]/30 bg-[#30d158]/[0.02]"
                            : isContradiction
                            ? "border-[#ff453a]/30 bg-[#ff453a]/[0.02]"
                            : "border-[#ffd60a]/30 bg-[#ffd60a]/[0.02]"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2 mb-2">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-mono font-bold text-zinc-400">
                              Claim #{i + 1}
                            </span>
                            <span
                              className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold border ${
                                isVerified
                                  ? "bg-[#30d158]/20 text-[#30d158] border-[#30d158]/30"
                                  : isContradiction
                                  ? "bg-[#ff453a]/20 text-[#ff453a] border-[#ff453a]/30"
                                  : "bg-[#ffd60a]/20 text-[#ffd60a] border-[#ffd60a]/30"
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
                              <span className="text-[10px] font-mono text-[#0a84ff] bg-[#0a84ff]/10 px-2 py-0.5 rounded-full border border-[#0a84ff]/20">
                                Confidence: {(c.confidence_score * 100).toFixed(0)}%
                              </span>
                            )}
                          </div>

                          {c.chunk_id && (
                            <button
                              onClick={() => setActiveChunkModal(c)}
                              className="inline-flex items-center gap-1 text-[11px] font-mono text-[#0a84ff] hover:text-white bg-[#0a84ff]/10 px-2.5 py-0.5 rounded-full border border-[#0a84ff]/20 transition-colors"
                            >
                              <FileText className="h-3 w-3" />
                              <span>Chunk: {c.chunk_id}</span>
                              <ExternalLink className="h-2.5 w-2.5 ml-0.5" />
                            </button>
                          )}
                        </div>

                        <p className="text-xs text-zinc-200 font-mono bg-black/50 p-3 rounded-xl border border-white/[0.06]">
                          &quot;{c.claim_text}&quot;
                        </p>
                      </div>
                    );
                  })
                ) : (
                  <p className="text-xs text-zinc-500 italic">No claim objects returned.</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Source Passage Modal */}
        {activeChunkModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
            <div className="relative w-full max-w-2xl rounded-3xl border border-white/15 bg-[#0c0c0e] p-6 shadow-2xl space-y-4">
              <div className="flex items-center justify-between border-b border-white/[0.08] pb-3">
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4 text-[#0a84ff]" />
                  <h3 className="text-xs font-bold text-white font-mono">
                    Ground-Truth Source Passage [{activeChunkModal.chunk_id}]
                  </h3>
                </div>
                <button
                  onClick={() => setActiveChunkModal(null)}
                  className="p-1 rounded-full hover:bg-white/10 text-zinc-400 hover:text-white"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              <div className="space-y-1.5">
                <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
                  Associated Claim:
                </span>
                <p className="text-xs font-mono text-[#0a84ff] bg-[#0a84ff]/10 p-3 rounded-xl border border-[#0a84ff]/20">
                  {activeChunkModal.claim_text}
                </p>
              </div>

              <div className="space-y-1.5">
                <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
                  Source Passage Text:
                </span>
                <div className="text-xs font-mono text-zinc-300 bg-black/60 p-4 rounded-xl border border-white/[0.08] max-h-56 overflow-y-auto leading-relaxed">
                  {activeChunkModal.source_text || "Source text passage details available in backend store."}
                </div>
              </div>

              <div className="flex justify-end pt-2">
                <Button size="sm" variant="outline" className="rounded-full text-xs" onClick={() => setActiveChunkModal(null)}>
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
