"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  BookOpen, Check, ChevronDown, CircleHelp, FileSearch, FileText, Layers3, LoaderCircle, PanelRight,
  Menu, Moon, Plus, RotateCw, ScanLine, Search, Send, ShieldCheck, Sparkles, Square, Sun,
  Trash2, UploadCloud, X,
} from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import {
  deleteDocument, getClaimStatus, getDocuments, getOptions, processDocument,
  streamQuery, uploadDocument, type ClaimWithSource, type DocumentItem, type SseEvent,
} from "@/lib/api";

const prompts = [
  { icon: ScanLine, title: "Summarize a document", question: "What are the key milestones in the project brief?" },
  { icon: Search, title: "Find a specific detail", question: "What is the budget for the pilot program?" },
  { icon: ShieldCheck, title: "Identify risks", question: "What dependencies could delay the pilot?" },
];

const stages = [
  { label: "Retrieve passages", icon: Search, event: "retrieval_started" },
  { label: "Draft answer", icon: Sparkles, event: "generation_started" },
  { label: "Verify claims", icon: ShieldCheck, event: "verification_started" },
  { label: "Correct claims", icon: RotateCw, event: "correction_triggered" },
] as const;

type Answer = Extract<SseEvent, { type: "final_answer" }>;
type Turn = { id: string; question: string; answer?: Answer; error?: string; stopped?: boolean };
type Options = Awaited<ReturnType<typeof getOptions>>;

function errorMessage(error: unknown, fallback: string) {
  return error instanceof Error && error.name !== "TypeError" ? error.message : fallback;
}

export default function Home() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [options, setOptions] = useState<Options>();
  const [libraryError, setLibraryError] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [sidebar, setSidebar] = useState(false);
  const [mobileInspector, setMobileInspector] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [dark, setDark] = useState(false);
  const [scope, setScope] = useState("");
  const [correction, setCorrection] = useState(true);
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [events, setEvents] = useState<SseEvent["type"][]>([]);
  const [pending, setPending] = useState(false);
  const [evidence, setEvidence] = useState<ClaimWithSource>();
  const [removeTarget, setRemoveTarget] = useState<DocumentItem>();
  const uploadInput = useRef<HTMLInputElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);
  const conversation = useRef<HTMLDivElement>(null);
  const request = useRef<AbortController | null>(null);
  const turnNumber = useRef(0);
  const ready = documents.filter(doc => doc.status === "ready");
  const processing = documents.some(doc => doc.status === "processing");
  const scopedDocument = documents.find(doc => doc.document_id === scope);
  const canQuery = Boolean(options && ready.length && (!scopedDocument || scopedDocument.status === "ready"));
  const latest = turns.at(-1);
  const local = options?.mode === "local";

  const refresh = useCallback(async () => {
    setDocuments(await getDocuments());
  }, []);

  const connect = useCallback(async () => {
    setLoading(true);
    setLibraryError("");
    try {
      const [nextOptions, nextDocs] = await Promise.all([getOptions(), getDocuments()]);
      setOptions(nextOptions);
      setDocuments(nextDocs);
    } catch {
      setOptions(undefined);
      setLibraryError("Could not connect to the document service.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const initialLoad = setTimeout(() => {
      setDark(localStorage.getItem("rag-theme") === "dark");
      void connect();
    }, 0);
    return () => clearTimeout(initialLoad);
  }, [connect]);
  useEffect(() => {
    if (!processing) return;
    let stopped = false;
    let timeout: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const nextDocs = await getDocuments();
        if (stopped) return;
        setDocuments(nextDocs);
        setLibraryError(current => current === "Lost connection while indexing. Retrying…" ? "" : current);
        if (!nextDocs.some(doc => doc.status === "processing")) return;
      } catch {
        if (!stopped) setLibraryError("Lost connection while indexing. Retrying…");
      }
      if (!stopped) timeout = setTimeout(poll, 1500);
    }
    timeout = setTimeout(poll, 900);
    return () => { stopped = true; clearTimeout(timeout); };
  }, [processing]);
  useEffect(() => {
    if (conversation.current) conversation.current.scrollTop = conversation.current.scrollHeight;
  }, [turns, pending]);
  useEffect(() => () => request.current?.abort(), []);
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  async function addDocument(getFile: () => Promise<File>) {
    if (busy || !options) return;
    setBusy(true);
    setLibraryError("");
    try {
      const file = await getFile();
      if (!file.name.toLowerCase().endsWith(".pdf")) throw new Error("Choose a PDF file.");
      if (file.size > options.max_upload_mb * 1024 * 1024) {
        throw new Error(`PDFs must be ${options.max_upload_mb} MB or smaller.`);
      }
      const doc = await uploadDocument(file);
      setScope(doc.document_id);
      setDocuments(current => [{ ...doc, current_stage: "extracting", status: "processing" }, ...current]);
      await processDocument(doc.document_id);
    } catch (error) {
      setLibraryError(errorMessage(error, "Upload failed. Check the backend connection."));
      await refresh().catch(() => {});
    } finally {
      setBusy(false);
      if (uploadInput.current) uploadInput.current.value = "";
    }
  }

  async function loadSample() {
    await addDocument(async () => {
      const response = await fetch("/project-brief.pdf");
      if (!response.ok) throw new Error("Could not load the example PDF.");
      return new File([await response.blob()], "Example project brief.pdf", { type: "application/pdf" });
    });
  }

  async function actOnDocument(doc: DocumentItem, remove = false) {
    setBusy(true);
    setLibraryError("");
    try {
      if (remove) {
        await deleteDocument(doc.document_id);
        if (scope === doc.document_id) setScope("");
        setRemoveTarget(undefined);
      } else {
        await processDocument(doc.document_id, doc.status === "failed");
      }
      await refresh();
    } catch (error) {
      setLibraryError(errorMessage(error, "The document operation failed. Try again."));
      setRemoveTarget(undefined);
    } finally {
      setBusy(false);
    }
  }

  async function ask() {
    const text = question.trim();
    if (!text || request.current || !canQuery) return;
    const controller = new AbortController();
    request.current = controller;
    const id = String(++turnNumber.current);
    const update = (result: Partial<Turn>) => setTurns(previous =>
      previous.map(turn => turn.id === id ? { ...turn, ...result } : turn),
    );
    setTurns(previous => [...previous, { id, question: text }]);
    setQuestion("");
    setPending(true);
    setEvents([]);
    setEvidence(undefined);

    try {
      await streamQuery({
        query: text, documentId: scopedDocument?.document_id, enableCorrection: correction,
        signal: controller.signal,
        onEvent(event) {
          setEvents(previous => [...previous, event.type]);
          if (event.type === "final_answer") {
            update({ answer: event });
          } else if (event.type === "error") {
            update({ error: event.detail });
          }
        },
      });
    } catch (error) {
      update(controller.signal.aborted
        ? { stopped: true }
        : { error: errorMessage(error, "Could not connect to the answer service. Try again.") });
    } finally {
      request.current = null;
      setPending(false);
    }
  }

  function toggleTheme() {
    setDark(!dark);
    localStorage.setItem("rag-theme", dark ? "light" : "dark");
  }

  function newQuery() {
    setTurns([]);
    setEvidence(undefined);
    setEvents([]);
    setQuestion("");
    setSidebar(false);
    setMobileInspector(false);
    composer.current?.focus();
  }

  const activeEvent = events.slice().reverse().find(event => stages.some(stage => stage.event === event));
  const activeLabel = stages.find(stage => stage.event === activeEvent)?.label;
  const verified = latest?.answer?.claims.filter(claim => getClaimStatus(claim) === "verified").length ?? 0;
  const visibleStages = local ? stages.slice(0, 1) : correction ? stages : stages.slice(0, 3);

  function chooseDocument(id: string) {
    setScope(id);
    setSidebar(false);
  }

  function openSource(claim: ClaimWithSource) {
    setEvidence(claim);
    if (window.matchMedia("(max-width: 1199px)").matches) setMobileInspector(true);
  }

  function renderLibrary() {
    return (
      <div className="library flex h-full min-h-0 flex-col gap-5 overflow-y-auto p-5">
        <div className="flex items-center gap-3">
          <span className="brand-mark"><Layers3 className="size-5" /></span>
          <div><strong className="block text-[1.1rem] tracking-tight">Groundwork</strong><span className="text-sm text-sidebar-muted">Document research</span></div>
        </div>
        <Button className="h-11 w-full justify-center bg-[#7354e8] text-white hover:bg-[#886bf5]" disabled={pending} onClick={newQuery}><Plus /> New query</Button>

        <div className="flex min-h-36 flex-1 flex-col gap-2">
          <div className="flex items-center justify-between px-2">
            <h2 className="text-sm font-semibold text-sidebar-muted">Document library</h2>
            <Button variant="ghost" size="icon" className="sidebar-icon" aria-label="Refresh documents" onClick={connect} disabled={loading}><RotateCw /></Button>
          </div>
          <Button variant="ghost" className={cn("library-row", !scope && "library-selected")} aria-pressed={!scope} onClick={() => chooseDocument("")}>
            <BookOpen /><span className="flex-1 text-left">All documents</span><span className="text-sidebar-muted">{ready.length}</span>
          </Button>
          <div className="min-h-0 flex-1 space-y-1 overflow-y-auto">
            {loading && documents.length === 0 && <Skeleton className="mx-2 h-10 bg-white/10" />}
            {!loading && !documents.length && <p className="px-2 py-3 text-sm leading-relaxed text-sidebar-muted">Your uploaded PDFs will appear here.</p>}
            {documents.map(doc => (
              <div key={doc.document_id} className={cn("rounded-xl", scope === doc.document_id && "library-selected")}>
                <div className="flex min-w-0 items-center pr-1">
                  <Button variant="ghost" className="library-row min-w-0 flex-1" aria-pressed={scope === doc.document_id} disabled={doc.status !== "ready"} onClick={() => chooseDocument(doc.document_id)}>
                    {doc.status === "processing" ? <LoaderCircle className="motion-safe:animate-spin" /> : <FileText />}
                    <span className="min-w-0 flex-1 text-left">
                      <span className="block truncate" title={doc.filename}>{doc.filename}</span>
                      <span className="block text-xs capitalize text-sidebar-muted">{doc.status === "ready" ? "Ready" : doc.current_stage ?? doc.status}</span>
                    </span>
                  </Button>
                  {doc.status !== "processing" && <Button variant="ghost" size="icon-sm" className="sidebar-icon" disabled={busy} aria-label={`Remove ${doc.filename}`} onClick={() => { setSidebar(false); setRemoveTarget(doc); }}><Trash2 /></Button>}
                </div>
                {doc.failure_reason && <p className="break-words px-3 pb-2 text-sm text-rose-300">{doc.failure_reason}</p>}
                {(doc.status === "uploaded" || (doc.status === "failed" && doc.retryable)) &&
                  <Button variant="link" className="text-violet-200" disabled={busy} onClick={() => actOnDocument(doc)}>{doc.status === "failed" ? "Retry indexing" : "Index document"}</Button>}
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-2">
          {libraryError && <p className="break-words text-sm text-rose-300" role="alert">{libraryError}</p>}
          <Button variant="outline" className={cn("upload-zone", dragging && "bg-white/15")} disabled={busy || !options}
            onClick={() => uploadInput.current?.click()}
            onDragOver={event => { event.preventDefault(); if (!busy && options) setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={event => {
              event.preventDefault();
              setDragging(false);
              const file = event.dataTransfer.files[0];
              if (file) void addDocument(async () => file);
            }}>
            {busy ? <LoaderCircle className="size-5 motion-safe:animate-spin" /> : <UploadCloud className="size-5" />}
            <span>{busy ? "Working…" : "Upload PDF"}<span className="block text-xs font-normal text-sidebar-muted">Up to {options?.max_upload_mb ?? 20} MB</span></span>
          </Button>
          <Button variant="ghost" className="sidebar-icon h-9 w-full" disabled={busy || !options} onClick={loadSample}>Try an example PDF</Button>
        </div>
        <div className="flex items-center gap-2 border-t border-white/10 pt-4 text-sm text-sidebar-muted">
          <span className={cn("size-2 rounded-full", options ? "bg-emerald-400" : "bg-amber-400")} />
          {options ? (local ? "Local passage search" : "Cloud verification") : loading ? "Connecting…" : "Service unavailable"}
        </div>
      </div>
    );
  }

  function renderInspector() {
    if (evidence) {
      const page = evidence.page_number;
      const end = evidence.page_number_end;
      const source = evidence.source_text;
      const offset = source?.indexOf(evidence.claim_text) ?? -1;
      return (
        <div className="space-y-4 p-5">
          <div className="flex items-start gap-3">
            <span className="feature-icon"><FileText /></span>
            <div className="min-w-0">
              <h3 className="break-words font-semibold">{evidence.filename || "Source document"}</h3>
              <p className="text-sm text-muted-foreground">{page != null ? `Page ${page}${end && end !== page ? `–${end}` : ""}` : "Page unavailable"}</p>
            </div>
          </div>
          {evidence.verification_status && <Badge variant="secondary" className="capitalize">{getClaimStatus(evidence).replaceAll("_", " ")}</Badge>}
          <blockquote className="whitespace-pre-wrap break-words rounded-xl border border-border bg-muted/45 p-4 text-sm leading-6">
            {source && offset >= 0
              ? <>{source.slice(0, offset)}<mark className="rounded-sm bg-primary/15 text-foreground">{evidence.claim_text}</mark>{source.slice(offset + evidence.claim_text.length)}</>
              : source || "No source passage was available for this claim."}
          </blockquote>
        </div>
      );
    }
    return (
      <div className="space-y-6 p-5">
        <section>
          <h3 className="mb-3 text-sm font-semibold text-muted-foreground">Pipeline</h3>
          <ol className="space-y-2">
            {visibleStages.map(({ label, icon: Icon, event }) => {
              const visited = events.includes(event);
              const active = pending && activeEvent === event;
              return (
                <li className={cn("flex items-center gap-3 rounded-xl px-2 py-1.5 text-sm text-muted-foreground", active && "bg-primary/8 text-primary")} key={event} aria-current={active ? "step" : undefined}>
                  <span className={cn("grid size-9 shrink-0 place-items-center rounded-lg border border-border bg-card", visited && "text-primary")}><Icon className="size-4" /></span>
                  <span className="flex-1">{label}</span>
                  {active ? <LoaderCircle className="size-4 motion-safe:animate-spin" /> : visited && !latest?.error && !latest?.stopped ? <Check className="size-4 text-primary" aria-label="Completed" /> : null}
                </li>
              );
            })}
          </ol>
        </section>
        <Card className="gap-2 border border-border bg-muted/45 p-4 ring-0">
          <h3 className="flex items-center gap-2 font-semibold"><CircleHelp className="size-4 text-primary" />{latest?.answer ? "Latest result" : local ? "How passage search works" : "How verification works"}</h3>
          <p className="text-sm leading-6 text-muted-foreground">{latest?.answer
            ? `${latest.answer.claims.filter(claim => claim.is_valid_source).length} sources cited${local ? "" : ` · ${verified} verified claims`} · ${(latest.answer.latency_ms / 1000).toFixed(1)}s`
            : local ? "Groundwork finds matching excerpts and links them to the original PDF page." : "Groundwork checks generated claims against retrieved passages so you can inspect their evidence."}</p>
        </Card>
      </div>
    );
  }

  return (
    <div className="workspace">
      <input ref={uploadInput} type="file" accept=".pdf,application/pdf" hidden onChange={event => {
        const file = event.currentTarget.files?.[0];
        if (file) void addDocument(async () => file);
      }} />
      <aside className="hidden min-h-0 lg:block" aria-label="Document library">{renderLibrary()}</aside>
      <Dialog open={sidebar} onOpenChange={setSidebar}>
        <DialogContent className="library" placement="left" showClose={false}>
          <DialogTitle className="sr-only">Document library</DialogTitle>
          {renderLibrary()}
          <Button variant="ghost" size="icon" className="sidebar-icon absolute top-4 right-3" aria-label="Close document library" onClick={() => setSidebar(false)}><X /></Button>
        </DialogContent>
      </Dialog>

      <main className="flex min-h-0 min-w-0 flex-col">
        <header className="flex h-16 shrink-0 items-center justify-between gap-2 border-b border-border bg-card px-3 sm:px-6">
          <div className="flex min-w-0 items-center gap-2">
            <Button variant="ghost" size="icon" className="lg:hidden" aria-label="Open document library" onClick={() => setSidebar(true)}><Menu /></Button>
            <span className="hidden text-sm text-muted-foreground sm:inline">Workspace</span><span className="hidden text-muted-foreground sm:inline">/</span>
            <h1 className="truncate text-sm font-semibold">Research</h1>
          </div>
          <div className="flex shrink-0 items-center gap-1.5 sm:gap-2">
            {options?.mode === "cloud" && <label className="flex items-center gap-1.5 text-sm"><input className="accent-primary" type="checkbox" checked={correction} disabled={pending} onChange={event => setCorrection(event.target.checked)} /> Self-correction</label>}
            {local && <Badge variant="secondary" className="hidden h-7 px-2.5 text-sm sm:inline-flex"><FileSearch /> Passage search</Badge>}
            <Button variant="ghost" size="icon" aria-label={dark ? "Switch to light theme" : "Switch to dark theme"} onClick={toggleTheme}>{dark ? <Sun /> : <Moon />}</Button>
            <Button variant="ghost" size="icon" className="min-[1200px]:hidden" aria-label="Open query details" onClick={() => setMobileInspector(true)}><PanelRight /></Button>
          </div>
        </header>

        <div className="flex min-h-0 flex-1 overflow-y-auto" ref={conversation} aria-label="Research conversation">
          {!turns.length ? (
            <section className="canvas my-auto space-y-5 py-5 sm:space-y-7 sm:py-8">
              <div className="hero relative isolate overflow-hidden rounded-2xl border border-violet-400/20 px-5 py-7 sm:px-8 sm:py-10">
                <div className="hero-art" aria-hidden="true">
                  <span className="hero-sheet hero-sheet-back" />
                  <span className="hero-sheet"><FileText className="size-9" /><span /><span /><span /></span>
                  <span className="hero-seal"><ShieldCheck className="size-6" /></span>
                </div>
                <div className="relative max-w-md">
                  <h2 className="text-[clamp(2rem,4.1vw,4rem)] leading-[1.02] font-semibold tracking-[-.065em]">Ask your<br />documents.</h2>
                  <p className="mt-4 max-w-[28ch] text-base leading-relaxed text-muted-foreground">Search your PDFs and see the evidence behind each answer.</p>
                </div>
              </div>
              <section aria-labelledby="prompt-heading">
                <h2 id="prompt-heading" className="mb-3 text-sm font-semibold">Try a question</h2>
                <div className="grid gap-2 sm:grid-cols-3 sm:gap-3">
                  {prompts.map(({ icon: Icon, title, question: example }) => (
                    <Button variant="outline" key={title} className="prompt-card" onClick={() => { setQuestion(example); composer.current?.focus(); }}>
                      <span className="feature-icon"><Icon /></span>
                      <span className="min-w-0"><strong className="block font-semibold">{title}</strong><span className="mt-1 block text-sm font-normal leading-snug text-muted-foreground">{example}</span></span>
                    </Button>
                  ))}
                </div>
              </section>
            </section>
          ) : (
            <div className="canvas space-y-6 py-7">
              {turns.map(turn => (
                <article className="flex flex-col gap-3" key={turn.id}>
                  <p className="max-w-[90%] self-end whitespace-pre-wrap break-words rounded-2xl rounded-br-sm bg-secondary px-4 py-3 text-secondary-foreground">{turn.question}</p>
                  {turn.answer && <Card className="gap-3 border border-border p-4 ring-0 sm:p-5">
                    <h2 className="flex items-center gap-2.5 text-base font-semibold"><span className="feature-icon"><Layers3 /></span>{turn.answer.final_status === "source_passages" ? "Source passages" : "Answer"}</h2>
                    {turn.answer.claims.length ? turn.answer.claims.map((claim, index) => (
                      <p className="break-words text-[0.9375rem] leading-relaxed" key={claim.claim_id}>{turn.answer?.final_status === "source_passages" ? <q>{claim.claim_text}</q> : claim.claim_text}{" "}
                        {claim.is_valid_source && <Button variant="secondary" size="xs" className="align-middle aria-pressed:bg-primary aria-pressed:text-primary-foreground" aria-label={`View source ${index + 1}`} aria-pressed={evidence?.claim_id === claim.claim_id} onClick={() => openSource(claim)}>{index + 1}</Button>}
                      </p>
                    )) : <p className="text-base">{turn.answer.final_answer_text}</p>}
                    {turn.answer.final_status !== "source_passages" && <Badge variant="outline" className="h-auto whitespace-normal capitalize">{turn.answer.final_status.replaceAll("_", " ")}{turn.answer.retry_count ? ` · ${turn.answer.retry_count} correction attempts` : ""}</Badge>}
                  </Card>}
                  {turn.error && <Alert variant="destructive"><AlertDescription>{turn.error} <Button variant="link" onClick={() => { setQuestion(turn.question); composer.current?.focus(); }}>Edit and retry</Button></AlertDescription></Alert>}
                  {turn.stopped && <p className="text-sm text-muted-foreground">Query stopped.</p>}
                </article>
              ))}
              {pending && <p className="flex items-center gap-2 text-sm text-primary" role="status"><LoaderCircle className="size-4 motion-safe:animate-spin" /> {activeLabel ?? "Connecting…"}</p>}
            </div>
          )}
        </div>

        <div className="canvas shrink-0 pt-2 pb-3">
          <form className="rounded-2xl border border-border bg-card p-2.5 shadow-sm focus-within:border-primary/70 sm:p-3" onSubmit={event => { event.preventDefault(); void ask(); }}>
            <textarea className="block w-full resize-none rounded-lg px-2 py-1.5 text-base outline-none placeholder:text-muted-foreground" ref={composer} rows={2} maxLength={4000} aria-label="Question" placeholder="Ask a question about your documents…" value={question} onChange={event => setQuestion(event.target.value)} onKeyDown={event => {
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                void ask();
              }
            }} />
            <div className="mt-2 flex items-center justify-between gap-3">
              <label className="flex max-w-[80%] min-w-0 items-center gap-2 rounded-lg border border-border bg-muted/60 px-2 py-1.5 text-sm">
                <BookOpen className="size-4 shrink-0 text-primary" />
                <select className="min-w-0 cursor-pointer appearance-none truncate bg-transparent" aria-label="Search scope" value={scopedDocument?.document_id ?? ""} onChange={event => setScope(event.target.value)}><option value="">All documents</option>{scopedDocument && scopedDocument.status !== "ready" && <option value={scopedDocument.document_id} disabled>{scopedDocument.filename} ({scopedDocument.status})</option>}{ready.map(doc => <option key={doc.document_id} value={doc.document_id}>{doc.filename}</option>)}</select>
                <ChevronDown className="pointer-events-none size-4 shrink-0 text-muted-foreground" />
              </label>
              {pending ? <Button variant="secondary" type="button" size="icon-lg" aria-label="Stop query" onClick={() => request.current?.abort()}><Square /></Button> :
                <Button type="submit" size="icon-lg" aria-label="Send question" disabled={!question.trim() || !canQuery}><Send /></Button>}
            </div>
          </form>
          <p className="mt-2 text-center text-xs text-muted-foreground">{scopedDocument && scopedDocument.status !== "ready" ? "Wait for indexing or select a different search scope." : !ready.length ? "Upload a PDF or try the example to start asking questions." : "Enter to send · Shift + Enter for a new line"}</p>
        </div>
      </main>

      <aside className="hidden min-h-0 overflow-y-auto border-l border-border bg-card min-[1200px]:block" aria-label="Query details">
        <div className="flex h-16 items-center justify-between border-b border-border px-5">
          <h2 className="font-semibold">{evidence ? "Source passage" : "Query details"}</h2>
          {evidence && <Button variant="ghost" size="icon" aria-label="Close source passage" onClick={() => setEvidence(undefined)}><X /></Button>}
        </div>
        {renderInspector()}
      </aside>
      <Dialog open={mobileInspector} onOpenChange={open => { setMobileInspector(open); if (!open) setEvidence(undefined); }}>
        <DialogContent className="p-0">
          <DialogTitle className="border-b border-border px-5 py-4 pr-12 font-semibold">{evidence ? "Source passage" : "Query details"}</DialogTitle>
          {renderInspector()}
        </DialogContent>
      </Dialog>

      <Dialog open={Boolean(removeTarget)} onOpenChange={open => { if (!open && !busy) setRemoveTarget(undefined); }}>
        <DialogContent showClose={false}>
          <DialogTitle className="text-lg font-semibold">Remove document?</DialogTitle>
          <DialogDescription className="mt-2 break-words text-sm text-muted-foreground">{removeTarget?.filename} will be deleted from this library.</DialogDescription>
          <div className="mt-6 flex justify-end gap-2">
            <Button variant="outline" autoFocus disabled={busy} onClick={() => setRemoveTarget(undefined)}>Cancel</Button>
            <Button variant="destructive" disabled={busy || !removeTarget} onClick={() => { if (removeTarget) void actOnDocument(removeTarget, true); }}>Remove</Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
