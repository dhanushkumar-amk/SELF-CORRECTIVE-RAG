"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  BookOpen, Check, ChevronDown, CircleHelp, FileSearch, FileText, Layers3, LoaderCircle,
  Menu, Moon, Plus, RotateCw, ScanLine, Search, Send, ShieldCheck, Sparkles, Square, Sun,
  Trash2, UploadCloud, X,
} from "lucide-react";
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
  const ready = documents.filter(doc => doc.status === "ready");
  const processing = documents.some(doc => doc.status === "processing");
  const selected = ready.find(doc => doc.document_id === scope);
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
    if (!text || request.current || !ready.length || !options) return;
    const controller = new AbortController();
    request.current = controller;
    const id = crypto.randomUUID();
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
        query: text, documentId: selected?.document_id, enableCorrection: correction,
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
    composer.current?.focus();
  }

  const activeStage = stages.reduce((last, stage, index) => events.includes(stage.event) ? index : last, -1);
  const verified = latest?.answer?.claims.filter(claim => getClaimStatus(claim) === "verified").length ?? 0;

  return (
    <div className={`workspace ${dark ? "dark-workspace" : ""}`}>
      {sidebar && <button className="scrim" aria-label="Close document library" onClick={() => setSidebar(false)} />}
      <aside className={`sidebar ${sidebar ? "sidebar-open" : ""}`}>
        <div className="brand"><span className="brand-mark"><Layers3 size={22} /></span><div><strong>Groundwork</strong><small>Document research</small></div></div>
        <button className="new-query" disabled={pending} onClick={newQuery}><Plus size={18} /> New query</button>

        <div className="library-heading"><h2>Library</h2><button className="icon-button" aria-label="Refresh documents" onClick={connect} disabled={loading}><RotateCw size={16} /></button></div>
        <button className={`library-all ${!scope ? "selected" : ""}`} onClick={() => { setScope(""); setSidebar(false); }}><BookOpen size={17} /> All documents <span>{ready.length}</span></button>
        <div className="document-list">
          {loading && <p className="quiet">Loading library…</p>}
          {!loading && !documents.length && <p className="quiet">Your uploaded PDFs will appear here.</p>}
          {documents.map(doc => (
            <div key={doc.document_id} className={`document ${scope === doc.document_id ? "selected" : ""}`}>
              <div className="document-main">
                <button className="document-select" disabled={doc.status !== "ready"} onClick={() => { setScope(doc.document_id); setSidebar(false); }}>
                  <FileText size={17} /><span><strong title={doc.filename}>{doc.filename}</strong>
                  <small>{doc.status === "ready" ? "Ready" : doc.current_stage ?? doc.status}</small></span>
                </button>
                {doc.status !== "processing" && <button className="icon-button" disabled={busy} aria-label={`Remove ${doc.filename}`} onClick={() => setRemoveTarget(doc)}><Trash2 size={15} /></button>}
              </div>
              {doc.failure_reason && <p className="document-error">{doc.failure_reason}</p>}
              {(doc.status === "uploaded" || (doc.status === "failed" && doc.retryable)) &&
                <button className="text-action" disabled={busy} onClick={() => actOnDocument(doc)}>{doc.status === "failed" ? "Retry indexing" : "Index document"}</button>}
            </div>
          ))}
        </div>

        <div className="upload-area">
          {libraryError && <p className="document-error" role="alert">{libraryError}</p>}
          <input ref={uploadInput} type="file" accept=".pdf,application/pdf" hidden onChange={event => {
            const file = event.currentTarget.files?.[0];
            if (file) void addDocument(async () => file);
          }} />
          <button className="upload-button" onClick={() => uploadInput.current?.click()} disabled={busy || !options}>
            {busy ? <LoaderCircle className="spin" size={19} /> : <UploadCloud size={19} />}
            <span>{busy ? "Working…" : "Upload PDF"}<small>Up to {options?.max_upload_mb ?? 20} MB</small></span>
          </button>
          <button className="sample-button" disabled={busy || !options} onClick={loadSample}>Try an example PDF</button>
        </div>
        <div className="sidebar-footer"><span className={`status-dot ${options ? "connected" : ""}`} /> {options ? (local ? "Local passage search" : "Cloud verification") : loading ? "Connecting…" : "Service unavailable"}</div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div><button className="icon-button mobile-menu" aria-label="Open document library" onClick={() => setSidebar(true)}><Menu size={21} /></button><span>Workspace</span><span className="breadcrumb-divider">/</span><strong>Research</strong></div>
          <div className="topbar-controls">{options?.mode === "cloud" && <label className="correction-toggle"><input type="checkbox" checked={correction} disabled={pending} onChange={event => setCorrection(event.target.checked)} /> Self-correction</label>}
          {local && <span className="mode-label"><FileSearch size={15} /> Passage search</span>}
          <button className="icon-button" aria-label={dark ? "Switch to light theme" : "Switch to dark theme"} onClick={toggleTheme}>{dark ? <Sun size={17} /> : <Moon size={17} />}</button></div>
        </header>

        <div className={`conversation ${turns.length ? "with-turns" : ""}`} ref={conversation} aria-label="Research conversation">
          {!turns.length ? (
            <section className="welcome">
              <div className="hero-mark" aria-hidden="true"><Layers3 size={31} /></div>
              <h1>Ask your documents.</h1>
              <p>Find answers in your PDFs and see the passages that support them.</p>
              <div className="prompt-grid">
                {prompts.map(({ icon: Icon, title, question: example }) => (
                  <button className="prompt" key={title} onClick={() => { setQuestion(example); composer.current?.focus(); }}>
                    <span className="prompt-icon"><Icon size={19} /></span><strong>{title}</strong><span>{example}</span>
                  </button>
                ))}
              </div>
            </section>
          ) : (
            <div className="turns">
              {turns.map(turn => (
                <article className="turn" key={turn.id}>
                  <p className="question">{turn.question}</p>
                  {turn.answer && <div className="answer">
                    <h2><span className="answer-mark"><Layers3 size={16} /></span>{turn.answer.final_status === "source_passages" ? "Source passages" : "Answer"}</h2>
                    {turn.answer.claims.length ? turn.answer.claims.map((claim, index) => (
                      <p key={claim.claim_id}>{turn.answer?.final_status === "source_passages" ? <q>{claim.claim_text}</q> : claim.claim_text}{" "}
                        <button className="citation" aria-label={`View source ${index + 1}`} aria-pressed={evidence?.claim_id === claim.claim_id} onClick={() => setEvidence(claim)}>{index + 1}</button>
                      </p>
                    )) : <p>{turn.answer.final_answer_text}</p>}
                    {turn.answer.final_status !== "source_passages" && <small className="answer-status">{turn.answer.final_status.replaceAll("_", " ")}{turn.answer.retry_count ? ` · ${turn.answer.retry_count} correction attempts` : ""}</small>}
                  </div>}
                  {turn.error && <p className="error-box" role="alert">{turn.error} <button className="text-action" onClick={() => { setQuestion(turn.question); composer.current?.focus(); }}>Edit and retry</button></p>}
                  {turn.stopped && <p className="quiet">Query stopped.</p>}
                </article>
              ))}
              {pending && <p className="working" role="status"><LoaderCircle className="spin" size={18} /> {activeStage >= 0 ? stages[activeStage].label : "Connecting…"}</p>}
            </div>
          )}
        </div>

        <div className="composer-area">
          <form className="composer" onSubmit={event => { event.preventDefault(); void ask(); }}>
            <textarea ref={composer} rows={2} maxLength={4000} aria-label="Question" placeholder="Ask a question about your documents…" value={question} onChange={event => setQuestion(event.target.value)} onKeyDown={event => {
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                void ask();
              }
            }} />
            <div className="composer-actions">
              <label className="scope"><BookOpen size={15} /><select aria-label="Search scope" value={selected?.document_id ?? ""} onChange={event => setScope(event.target.value)}><option value="">All documents</option>{ready.map(doc => <option key={doc.document_id} value={doc.document_id}>{doc.filename}</option>)}</select><ChevronDown size={14} /></label>
              {pending ? <button className="send" type="button" aria-label="Stop query" onClick={() => request.current?.abort()}><Square size={17} /></button> :
                <button className="send" type="submit" aria-label="Send question" disabled={!question.trim() || !ready.length || !options}><Send size={18} /></button>}
            </div>
          </form>
          <p className="composer-note">{!ready.length ? "Upload a PDF or try the example to start asking questions." : "Press Enter to send · Shift + Enter for a new line"}</p>
        </div>
      </main>

      <aside className={`inspector ${evidence ? "evidence-open" : ""}`} aria-label="Query details">
        <div className="inspector-header"><h2>{evidence ? "Source passage" : "Query details"}</h2>{evidence && <button className="icon-button" aria-label="Close source passage" onClick={() => setEvidence(undefined)}><X size={19} /></button>}</div>
        {evidence ? (
          <div className="evidence">
            <span className="prompt-icon"><FileText size={20} /></span>
            <h3>{evidence.filename || "Source document"}</h3>
            <p className="quiet">{evidence.page_number != null ? `Page ${evidence.page_number}${evidence.page_number_end && evidence.page_number_end !== evidence.page_number ? `–${evidence.page_number_end}` : ""}` : "Page unavailable"}{evidence.verification_status ? ` · ${getClaimStatus(evidence).replaceAll("_", " ")}` : ""}</p>
            <blockquote>{evidence.source_text || "No source passage was available for this claim."}</blockquote>
          </div>
        ) : (
          <>
            <div className="pipeline"><h3>Pipeline</h3>
              {(local ? stages.slice(0, 1) : correction ? stages : stages.slice(0, 3)).map(({ label, icon: Icon, event }, index) => {
                const visited = events.includes(event);
                const active = pending && visited && activeStage === index;
                return <div className={`stage ${visited ? "stage-visited" : ""}`} key={label}><span><Icon size={17} /></span><span>{label}</span>{active ? <LoaderCircle className="spin stage-indicator" size={15} /> : visited && !latest?.error && !latest?.stopped ? <Check className="stage-indicator" size={16} /> : null}</div>;
              })}
            </div>
            <div className="inspector-note"><CircleHelp size={18} /><div><strong>{latest?.answer ? "Result" : "About this workspace"}</strong><p>{latest?.answer
              ? `${latest.answer.claims.length} cited passages${local ? "" : ` · ${verified} verified claims`} · ${(latest.answer.latency_ms / 1000).toFixed(1)}s`
              : local ? "Local search returns excerpts from your PDFs with filename and page citations." : "Source citations let you inspect the evidence behind each claim."}</p></div></div>
          </>
        )}
      </aside>

      {removeTarget && <dialog open className="confirm" onKeyDown={event => { if (event.key === "Escape" && !busy) setRemoveTarget(undefined); }}>
        <h2>Remove document?</h2><p>{removeTarget.filename} will be deleted from this library.</p>
        <div><button className="sample-button" autoFocus disabled={busy} onClick={() => setRemoveTarget(undefined)}>Cancel</button><button className="danger" disabled={busy} onClick={() => actOnDocument(removeTarget, true)}>Remove</button></div>
      </dialog>}
    </div>
  );
}
