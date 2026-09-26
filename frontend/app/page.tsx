"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Send,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  FileText,
  HelpCircle,
  UploadCloud,
  Loader2,
  Trash2,
  Layers,
  Plus,
  Sun,
  Moon,
  PanelLeftClose,
  PanelLeft,
  BarChart3,
  User,
  Bot,
  Zap,
  Activity,
  WifiOff,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { ClaimBadge } from "@/components/ClaimBadge";
import {
  API_URL,
  ApiError,
  deleteDocument,
  getClaimStatus,
  getDocuments,
  processDocument,
  streamQuery,
  uploadDocument,
  type ClaimWithSource,
  type DocumentItem,
  type SseEvent,
} from "@/lib/api";
import { cn } from "@/lib/utils";

/* ═══════════════════════════════════════════
   Constants
   ═══════════════════════════════════════════ */

const SAMPLE_PROMPTS = [
  "What were Q3 revenues and key growth metrics?",
  "Is the deployment process fully automated?",
  "Summarize the risk factors mentioned in the report.",
];

const PIPELINE_STEPS = ["retrieving", "generating", "verifying", "correcting"] as const;
type PipelineStep = (typeof PIPELINE_STEPS)[number] | "done" | "error";

const STEP_LABELS: Record<PipelineStep, string> = {
  retrieving: "Retrieving relevant information…",
  generating: "Generating answer…",
  verifying: "Verifying claims…",
  correcting: "Found an issue, double-checking…",
  done: "Answer ready",
  error: "Something went wrong",
};

/* ═══════════════════════════════════════════
   Types
   ═══════════════════════════════════════════ */

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  claims?: ClaimWithSource[];
  finalStatus?: string;
  correctionTriggered?: boolean;
  executionTimeMs?: number;
  error?: { code: string; detail: string };
}

/* ═══════════════════════════════════════════
   Theme Hook
   ═══════════════════════════════════════════ */

function useTheme() {
  const [theme, setThemeState] = useState<"dark" | "light">(() => {
    if (typeof window === "undefined") return "dark";
    const saved = localStorage.getItem("rag-theme") as "dark" | "light" | null;
    return saved ?? "dark";
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  const toggle = useCallback(() => {
    setThemeState((current) => {
      const next = current === "dark" ? "light" : "dark";
      localStorage.setItem("rag-theme", next);
      return next;
    });
  }, []);

  return { theme, toggle };
}

/* ═══════════════════════════════════════════
   Small helpers
   ═══════════════════════════════════════════ */

function docStatusColor(status: DocumentItem["status"]) {
  if (status === "ready") return "bg-emerald-500";
  if (status === "failed") return "bg-red-500";
  return "bg-amber-500 animate-pulse";
}

/* ═══════════════════════════════════════════
   Main Page Component
   ═══════════════════════════════════════════ */

export default function Home() {
  const { theme, toggle: toggleTheme } = useTheme();

  // Layout
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(() =>
    typeof window === "undefined" ? true : window.matchMedia("(min-width: 768px)").matches,
  );

  // Documents
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [selectedDocId, setSelectedDocId] = useState<string | undefined>();
  const [uploading, setUploading] = useState(false);
  const [docError, setDocError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Settings
  const [enableCorrection, setEnableCorrection] = useState(true);

  // Chat
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [pipelineStep, setPipelineStep] = useState<PipelineStep>("retrieving");
  const [logs, setLogs] = useState<string[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // ─── Documents ───
  const fetchDocs = useCallback(async () => {
    setLoadingDocs(true);
    setDocError(null);
    try {
      setDocuments(await getDocuments());
    } catch (err) {
      setDocError(err instanceof ApiError ? err.message : "Could not load documents.");
    } finally {
      setLoadingDocs(false);
    }
  }, []);

  useEffect(() => {
    fetchDocs();
  }, [fetchDocs]);

  const handleUpload = async (file: File) => {
    setUploading(true);
    setDocError(null);
    try {
      const uploaded = await uploadDocument(file);
      await processDocument(uploaded.document_id);
      await fetchDocs();
      setSelectedDocId(uploaded.document_id);
    } catch (err) {
      setDocError(
        err instanceof ApiError ? err.message : "Upload failed. Is the backend running?",
      );
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteDocument(id);
    } catch {
      /* deletion is best-effort; refresh shows remaining state */
    }
    if (selectedDocId === id) setSelectedDocId(undefined);
    fetchDocs();
  };

  // ─── Chat ───
  const newChat = () => {
    setMessages([]);
    setLogs([]);
    setPipelineStep("retrieving");
  };

  const sendQuery = async (text?: string) => {
    const q = (text ?? input).trim();
    if (!q || streaming) return;

    setInput("");
    setStreaming(true);
    setPipelineStep("retrieving");
    setLogs(["Connecting to pipeline…"]);

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: q };
    const assistantId = (Date.now() + 1).toString();
    setMessages((prev) => [...prev, userMsg]);

    let correctionTriggered = false;

    const handleEvent = (event: SseEvent) => {
      switch (event.type) {
        case "retrieval_started":
          setPipelineStep("retrieving");
          setLogs((prev) => [...prev, "Hybrid Pinecone + BM25 search…"]);
          break;
        case "retrieval_complete":
          setLogs((prev) => [
            ...prev,
            `Retrieved ${event.chunk_count} relevant passage${event.chunk_count === 1 ? "" : "s"}`,
          ]);
          break;
        case "generation_started":
          setPipelineStep("generating");
          setLogs((prev) => [...prev, "Generating candidate answer…"]);
          break;
        case "generation_complete":
          setLogs((prev) => [...prev, `Extracted ${event.claim_count} claims`]);
          break;
        case "verification_started":
          setPipelineStep("verifying");
          setLogs((prev) => [...prev, "DeBERTa-v3 NLI verification…"]);
          break;
        case "verification_complete":
          setLogs((prev) => [...prev, `Verification pass complete (${event.claims.length} claims)`]);
          break;
        case "correction_triggered":
          correctionTriggered = true;
          setPipelineStep("correcting");
          setLogs((prev) => [
            ...prev,
            `Correction loop triggered — ${event.failed_claims.length} claim(s) flagged`,
          ]);
          break;
        case "final_answer":
          setPipelineStep("done");
          setLogs((prev) => [...prev, "Complete ✓"]);
          setMessages((prev) => [
            ...prev,
            {
              id: assistantId,
              role: "assistant",
              content: event.final_answer_text,
              claims: event.claims,
              finalStatus: event.final_status,
              correctionTriggered,
              executionTimeMs: event.latency_ms,
            },
          ]);
          break;
        case "error":
          setPipelineStep("error");
          setLogs((prev) => [...prev, `Pipeline error: ${event.error}`]);
          setMessages((prev) => [
            ...prev,
            {
              id: assistantId,
              role: "assistant",
              content: "",
              error: { code: event.error, detail: event.detail },
            },
          ]);
          break;
      }
    };

    try {
      await streamQuery({
        query: q,
        documentId: selectedDocId ?? null,
        enableCorrection,
        onEvent: handleEvent,
      });
    } catch (err) {
      setPipelineStep("error");
      if (err instanceof ApiError) {
        setMessages((prev) => [
          ...prev,
          { id: assistantId, role: "assistant", content: "", error: { code: err.code, detail: err.message } },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            id: assistantId,
            role: "assistant",
            content: "",
            error: {
              code: "CONNECTION_ERROR",
              detail: `Cannot reach the backend at ${API_URL}. Make sure the FastAPI server is running.`,
            },
          },
        ]);
      }
    } finally {
      setStreaming(false);
    }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, logs]);

  /* ═══════════════════════════════════════════
     Summary badge
     ═══════════════════════════════════════════ */

  const summaryFor = (msg: Message) => {
    if (!msg.claims || msg.claims.length === 0) return null;
    const verified = msg.claims.filter((c) => getClaimStatus(c) === "verified").length;
    const total = msg.claims.length;
    const status = msg.finalStatus ?? "unverifiable";

    const style =
      status === "fully_verified"
        ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
        : status.startsWith("partially_verified")
          ? "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20"
          : "bg-zinc-500/10 text-zinc-500 border-zinc-500/20 dark:text-zinc-400";

    const icon =
      status === "fully_verified" ? (
        <CheckCircle2 className="h-3 w-3" />
      ) : (
        <AlertTriangle className="h-3 w-3" />
      );

    const label =
      status === "fully_verified"
        ? "Fully Verified"
        : status.startsWith("partially_verified")
          ? "Partially Verified"
          : "Unverifiable";

    return { style, icon, label, verified, total };
  };

  /* ═══════════════════════════════════════════
     RENDER
     ═══════════════════════════════════════════ */

  return (
    <div className="flex h-full overflow-hidden bg-background text-foreground transition-colors duration-200">
      {/* ═══════ SIDEBAR ═══════ */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex h-full w-[280px] shrink-0 flex-col border-r border-border bg-card",
          "transform transition-transform duration-200 md:static md:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        {/* Sidebar Header */}
        <div className="flex h-14 items-center justify-between border-b border-border px-4">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-emerald-500" />
            <span className="text-[13px] font-semibold tracking-tight text-foreground">
              Self-Correcting RAG
            </span>
          </div>
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={() => setSidebarOpen(false)}
            className="text-muted-foreground hover:text-foreground md:hidden"
          >
            <PanelLeftClose className="h-4 w-4" />
          </Button>
        </div>

        {/* New Chat + Upload */}
        <div className="space-y-2 border-b border-border p-3">
          <Button
            variant="default"
            size="sm"
            onClick={newChat}
            className="h-9 w-full justify-start text-[13px]"
          >
            <Plus className="mr-2 h-3.5 w-3.5" /> New Query
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
            className="h-9 w-full justify-start text-[13px]"
          >
            {uploading ? (
              <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
            ) : (
              <UploadCloud className="mr-2 h-3.5 w-3.5" />
            )}
            {uploading ? "Processing…" : "Upload PDF"}
          </Button>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.txt"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleUpload(f);
            }}
          />
          {docError && (
            <p className="text-[11px] leading-snug text-destructive">{docError}</p>
          )}
        </div>

        {/* Settings */}
        <div className="space-y-3 border-b border-border px-4 py-3">
          <div className="flex items-center justify-between">
            <span className="text-[12px] text-muted-foreground">NLI Correction Loop</span>
            <button
              onClick={() => setEnableCorrection(!enableCorrection)}
              className={cn(
                "relative h-[18px] w-8 rounded-full transition-colors",
                enableCorrection ? "bg-emerald-500" : "bg-muted-foreground/30",
              )}
              aria-label="Toggle correction loop"
            >
              <span
                className={cn(
                  "absolute top-[2px] h-[14px] w-[14px] rounded-full bg-white shadow-sm transition-transform",
                  enableCorrection ? "left-[16px]" : "left-[2px]",
                )}
              />
            </button>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[12px] text-muted-foreground">Appearance</span>
            <Button
              variant="ghost"
              size="icon-xs"
              onClick={toggleTheme}
              className="text-muted-foreground hover:text-foreground"
            >
              {theme === "dark" ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
            </Button>
          </div>
        </div>

        {/* Documents */}
        <div className="flex-1 overflow-y-auto">
          <div className="flex items-center justify-between px-4 pb-1 pt-3">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              Documents
            </span>
            <button onClick={fetchDocs} className="text-muted-foreground hover:text-foreground">
              <RefreshCw className={cn("h-3 w-3", loadingDocs && "animate-spin")} />
            </button>
          </div>

          <div className="space-y-1 px-3 pb-3">
            <button
              onClick={() => setSelectedDocId(undefined)}
              className={cn(
                "w-full rounded-lg px-3 py-2 text-left text-[12px] transition-colors",
                !selectedDocId
                  ? "bg-accent font-medium text-foreground"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
            >
              All Documents
            </button>

            {loadingDocs ? (
              <div className="space-y-2 px-1 py-3">
                <Skeleton className="h-9 w-full" />
                <Skeleton className="h-9 w-full" />
              </div>
            ) : documents.length === 0 ? (
              <div className="py-6 text-center text-[12px] text-muted-foreground">
                <Layers className="mx-auto mb-1.5 h-5 w-5 opacity-40" />
                No documents yet — upload a PDF to get started
              </div>
            ) : (
              documents.map((doc) => (
                <div
                  key={doc.document_id}
                  onClick={() => setSelectedDocId(doc.document_id)}
                  className={cn(
                    "group flex cursor-pointer items-center justify-between rounded-lg px-3 py-2.5 transition-colors",
                    selectedDocId === doc.document_id
                      ? "bg-accent text-foreground"
                      : "text-muted-foreground hover:bg-accent hover:text-foreground",
                  )}
                >
                  <div className="flex min-w-0 items-center gap-2">
                    <FileText className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                    <div className="min-w-0">
                      <div className="truncate text-[12px] font-medium">{doc.filename}</div>
                      <div className="text-[10px] text-muted-foreground">
                        {doc.chunk_count ?? 0} chunks · {doc.status}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className={cn("h-1.5 w-1.5 rounded-full", docStatusColor(doc.status))} />
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(doc.document_id);
                      }}
                      className="text-muted-foreground opacity-0 transition-opacity hover:text-red-500 group-hover:opacity-100"
                      aria-label={`Delete ${doc.filename}`}
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Sidebar Footer */}
        <div className="border-t border-border px-4 py-3 text-center text-[11px] text-muted-foreground">
          DeBERTa-v3 NLI · LangGraph · FastAPI
        </div>
      </aside>

      {/* ═══════ MAIN CHAT AREA ═══════ */}
      <div className="flex h-full min-w-0 flex-1 flex-col">
        {/* Top Bar */}
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-background px-4">
          <div className="flex items-center gap-3">
            {!sidebarOpen && (
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => setSidebarOpen(true)}
                className="text-muted-foreground"
              >
                <PanelLeft className="h-4 w-4" />
              </Button>
            )}
            <div className="flex items-center gap-2">
              {!sidebarOpen && <ShieldCheck className="h-4 w-4 text-emerald-500" />}
              <span className="text-[13px] font-medium text-foreground">
                {selectedDocId
                  ? `Querying: ${documents.find((d) => d.document_id === selectedDocId)?.filename ?? selectedDocId.slice(0, 12) + "…"}`
                  : "All Documents"}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {enableCorrection && (
              <span className="rounded border border-emerald-500/20 bg-emerald-500/10 px-2 py-0.5 text-[11px] font-medium text-emerald-600 dark:text-emerald-400">
                NLI Active
              </span>
            )}
            {!sidebarOpen && (
              <Button variant="ghost" size="icon-xs" onClick={toggleTheme} className="text-muted-foreground">
                {theme === "dark" ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
              </Button>
            )}
          </div>
        </header>

        {/* Message Scroll Area */}
        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
            {/* Empty State */}
            {messages.length === 0 && !streaming && (
              <div className="flex flex-col items-center justify-center pb-12 pt-20 text-center">
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl border border-border bg-card">
                  <ShieldCheck className="h-6 w-6 text-emerald-500" />
                </div>
                <h2 className="mb-1 text-lg font-semibold text-foreground">Self-Correcting RAG</h2>
                <p className="max-w-sm text-[13px] text-muted-foreground">
                  Ask a question about your documents. Every claim is verified for factual
                  accuracy using DeBERTa-v3 NLI — and corrected automatically when it fails.
                </p>
                <div className="mt-6 flex flex-wrap justify-center gap-2">
                  {SAMPLE_PROMPTS.map((p, i) => (
                    <button
                      key={i}
                      onClick={() => sendQuery(p)}
                      className="max-w-[260px] rounded-lg border border-border bg-card px-3.5 py-2 text-left text-[12px] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Messages */}
            {messages.map((msg) => {
              if (msg.role === "user") {
                return (
                  <div key={msg.id} className="flex justify-end gap-3">
                    <div className="order-first max-w-[85%] min-w-0">
                      <div className="rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-[14px] leading-relaxed text-primary-foreground">
                        {msg.content}
                      </div>
                    </div>
                    <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-primary">
                      <User className="h-3.5 w-3.5 text-primary-foreground" />
                    </div>
                  </div>
                );
              }

              // ── Error message ──
              if (msg.error) {
                const rateLimited = msg.error.code === "RATE_LIMIT_EXCEEDED";
                const generationFailed = msg.error.code === "GENERATION_ERROR";
                return (
                  <div key={msg.id} className="flex gap-3">
                    <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-border bg-card">
                      <Bot className="h-3.5 w-3.5 text-foreground" />
                    </div>
                    <Alert variant="destructive" className="max-w-[85%] min-w-0">
                      <WifiOff className="h-4 w-4" />
                      <AlertTitle>
                        {rateLimited
                          ? "Rate limit exceeded"
                          : generationFailed
                            ? "Answer generation failed"
                            : "Request failed"}
                      </AlertTitle>
                      <AlertDescription>{msg.error.detail}</AlertDescription>
                    </Alert>
                  </div>
                );
              }

              // ── No relevant information ──
              const noResults =
                (!msg.claims || msg.claims.length === 0) &&
                msg.finalStatus === "unverifiable";

              const summary = summaryFor(msg);

              return (
                <div key={msg.id} className="flex gap-3">
                  <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-border bg-card">
                    <Bot className="h-3.5 w-3.5 text-foreground" />
                  </div>

                  <div className="max-w-[85%] min-w-0 flex-1 space-y-3">
                    {/* Status header + summary */}
                    <div className="flex flex-wrap items-center gap-2">
                      {noResults ? (
                        <span className="inline-flex items-center gap-1 rounded-md border border-zinc-500/20 bg-zinc-500/10 px-2 py-0.5 text-[11px] font-medium text-zinc-500 dark:text-zinc-400">
                          <HelpCircle className="h-3 w-3" /> No Relevant Information
                        </span>
                      ) : summary ? (
                        <>
                          <span
                            className={cn(
                              "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[11px] font-medium",
                              summary.style,
                            )}
                          >
                            {summary.icon} {summary.label}
                          </span>
                          <span className="text-[11px] text-muted-foreground">
                            {summary.verified}/{summary.total} claims verified
                          </span>
                        </>
                      ) : null}

                      {msg.executionTimeMs != null && msg.executionTimeMs > 0 && (
                        <span className="flex items-center gap-1 text-[11px] text-muted-foreground">
                          <Activity className="h-3 w-3" /> {msg.executionTimeMs}ms
                        </span>
                      )}
                      {msg.correctionTriggered && (
                        <span className="inline-flex items-center gap-1 rounded-md border border-amber-500/20 bg-amber-500/10 px-2 py-0.5 text-[11px] font-medium text-amber-600 dark:text-amber-400">
                          <RefreshCw className="h-3 w-3" /> Correction Applied
                        </span>
                      )}
                    </div>

                    {noResults ? (
                      <Alert>
                        <FileText className="h-4 w-4" />
                        <AlertTitle>No relevant information found</AlertTitle>
                        <AlertDescription>
                          None of your ingested documents contain passages relevant enough to
                          answer this question. Try rephrasing, or query a different document.
                        </AlertDescription>
                      </Alert>
                    ) : (
                      <>
                        {/* Answer text */}
                        {msg.content && (
                          <div className="text-[14px] leading-relaxed text-foreground">
                            {msg.content}
                          </div>
                        )}

                        {/* Claims breakdown */}
                        {msg.claims && msg.claims.length > 0 && (
                          <div className="overflow-hidden rounded-xl border border-border bg-card">
                            <div className="flex items-center justify-between border-b border-border bg-muted/30 px-3.5 py-2.5">
                              <span className="flex items-center gap-1.5 text-[11px] font-semibold text-foreground">
                                <BarChart3 className="h-3 w-3 text-muted-foreground" />
                                {msg.claims.length} Verified Claim{msg.claims.length === 1 ? "" : "s"}
                              </span>
                              <div className="flex items-center gap-1.5">
                                {msg.claims.map((c, j) => (
                                  <span
                                    key={j}
                                    className={cn(
                                      "h-1.5 w-1.5 rounded-full",
                                      getClaimStatus(c) === "verified"
                                        ? "bg-emerald-500"
                                        : getClaimStatus(c) === "contradicted"
                                          ? "bg-red-500"
                                          : getClaimStatus(c) === "unverifiable"
                                            ? "bg-zinc-500"
                                            : "bg-amber-500",
                                    )}
                                  />
                                ))}
                              </div>
                            </div>

                            <div className="divide-y divide-border">
                              {msg.claims.map((claim) => (
                                <div key={claim.claim_id} className="space-y-1.5 px-3.5 py-3">
                                  <div className="flex items-center justify-between gap-2">
                                    <ClaimBadge claim={claim} />
                                    {claim.page_number != null && (
                                      <span className="shrink-0 text-[11px] text-muted-foreground">
                                        {claim.filename ? `${claim.filename} · ` : ""}p.{" "}
                                        {claim.page_number}
                                        {claim.page_number_end && claim.page_number_end !== claim.page_number
                                          ? `–${claim.page_number_end}`
                                          : ""}
                                      </span>
                                    )}
                                  </div>
                                  <p className="text-[13px] leading-snug text-foreground/80">
                                    {claim.claim_text}
                                  </p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              );
            })}

            {/* Streaming pipeline indicator */}
            {streaming && (
              <div className="flex gap-3">
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-border bg-card">
                  <Loader2 className="h-3.5 w-3.5 animate-spin text-foreground" />
                </div>
                <div className="max-w-md flex-1 space-y-3 rounded-xl border border-border bg-card p-4">
                  {/* Pipeline Progress */}
                  <div className="flex items-center gap-1.5">
                    {PIPELINE_STEPS.map((step) => {
                      const stepIdx = (PIPELINE_STEPS as readonly string[]).indexOf(step);
                      const currentIdx = (PIPELINE_STEPS as readonly string[]).indexOf(pipelineStep);
                      const active = pipelineStep === step;
                      const passed = currentIdx > stepIdx;
                      return (
                        <div
                          key={step}
                          className={cn(
                            "h-1 flex-1 rounded-full transition-colors",
                            active
                              ? "animate-pulse bg-foreground"
                              : passed
                                ? "bg-emerald-500"
                                : "bg-muted",
                          )}
                        />
                      );
                    })}
                  </div>
                  <div className="flex items-center gap-2">
                    <Zap className="h-3 w-3 text-muted-foreground" />
                    <span className="text-[12px] text-muted-foreground">
                      {STEP_LABELS[pipelineStep]}
                    </span>
                  </div>
                  <div className="max-h-20 space-y-0.5 overflow-y-auto font-mono text-[11px] text-muted-foreground">
                    {logs.map((l, i) => (
                      <div key={i}>› {l}</div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>
        </div>

        {/* Input Bar */}
        <div className="shrink-0 border-t border-border bg-background px-4 py-3">
          <div className="mx-auto max-w-3xl">
            {messages.length > 0 && !streaming && (
              <div className="mb-2 flex gap-1.5 overflow-x-auto pb-1">
                {SAMPLE_PROMPTS.map((p, i) => (
                  <button
                    key={i}
                    onClick={() => sendQuery(p)}
                    className="shrink-0 whitespace-nowrap rounded-md border border-border bg-card px-2.5 py-1 text-[11px] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                  >
                    {p}
                  </button>
                ))}
              </div>
            )}

            <div className="flex items-center gap-2 rounded-xl border border-border bg-card shadow-sm transition-all focus-within:border-foreground/20 focus-within:ring-2 focus-within:ring-ring/20">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendQuery();
                  }
                }}
                placeholder="Ask a question about your documents…"
                className="flex-1 bg-transparent py-3 pl-4 text-[14px] text-foreground placeholder:text-muted-foreground focus:outline-none"
                disabled={streaming}
              />
              <Button
                variant="default"
                size="icon-sm"
                onClick={() => sendQuery()}
                disabled={streaming || !input.trim()}
                className="mr-2 h-8 w-8 rounded-lg"
                aria-label="Send query"
              >
                {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </Button>
            </div>

            <p className="mt-2 text-center text-[11px] text-muted-foreground">
              Responses are verified using DeBERTa-v3 NLI. Claims are automatically scored for
              factual accuracy — {enableCorrection ? "and corrected when they fail." : "correction loop is off."}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
