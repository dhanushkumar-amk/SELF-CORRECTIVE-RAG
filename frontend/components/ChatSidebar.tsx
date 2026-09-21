"use client";

import { useEffect, useState, useRef } from "react";
import {
  FileText,
  UploadCloud,
  CheckCircle,
  AlertCircle,
  Loader2,
  Trash2,
  RefreshCw,
  Layers,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
  Plus,
  BarChart2,
  SlidersHorizontal,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/ThemeToggle";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface DocumentItem {
  document_id: string;
  filename: string;
  file_size_bytes?: number;
  status: "UPLOADED" | "PROCESSING" | "PROCESSED" | "FAILED";
  total_chunks?: number;
  created_at?: string;
  error_message?: string;
}

interface ChatSidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  selectedDocumentId?: string;
  onSelectDocument: (docId?: string) => void;
  enableCorrection: boolean;
  onToggleCorrection: () => void;
  showAnalytics: boolean;
  onToggleAnalytics: () => void;
  onNewChat: () => void;
}

export function ChatSidebar({
  isOpen,
  onToggle,
  selectedDocumentId,
  onSelectDocument,
  enableCorrection,
  onToggleCorrection,
  showAnalytics,
  onToggleAnalytics,
  onNewChat,
}: ChatSidebarProps) {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [loadingDocs, setLoadingDocs] = useState<boolean>(true);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDocuments = async () => {
    setLoadingDocs(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/v1/documents`, { cache: "no-store" });
      if (res.ok) {
        const data = await res.json();
        const list = Array.isArray(data) ? data : data.documents || [];
        setDocuments(list);
      } else {
        setError("Failed to fetch documents");
      }
    } catch (err: any) {
      setError(err?.message || "Backend offline");
    } finally {
      setLoadingDocs(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleFileUpload = async (file: File) => {
    if (!file) return;
    setUploading(true);
    setUploadProgress(`Uploading ${file.name}...`);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const uploadRes = await fetch(`${API_URL}/api/v1/documents/upload`, {
        method: "POST",
        body: formData,
      });

      if (!uploadRes.ok) {
        const errJson = await uploadRes.json().catch(() => ({}));
        throw new Error(errJson.detail || errJson.message || "Failed to upload");
      }

      const uploadData = await uploadRes.json();
      const docId = uploadData.document_id || uploadData.id;

      setUploadProgress(`Processing chunks...`);
      const processRes = await fetch(`${API_URL}/api/v1/documents/${docId}/process`, {
        method: "POST",
      });

      if (!processRes.ok) {
        const errJson = await processRes.json().catch(() => ({}));
        throw new Error(errJson.detail || errJson.message || "Failed to process");
      }

      await fetchDocuments();
      onSelectDocument(docId);
    } catch (err: any) {
      setError(err?.message || "Upload failed");
    } finally {
      setUploading(false);
      setUploadProgress("");
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDelete = async (docId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm("Delete document from index?")) return;

    try {
      const res = await fetch(`${API_URL}/api/v1/documents/${docId}`, { method: "DELETE" });
      if (res.ok) {
        if (selectedDocumentId === docId) onSelectDocument(undefined);
        await fetchDocuments();
      }
    } catch {
      alert("Failed to delete document");
    }
  };

  if (!isOpen) {
    return (
      <div className="fixed top-3 left-3 z-40">
        <Button
          variant="outline"
          size="icon-sm"
          onClick={onToggle}
          title="Open Knowledge Base Sidebar"
          className="border-border bg-background shadow-sm"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <aside className="w-72 border-r border-border bg-card flex flex-col h-screen shrink-0 transition-all z-30 font-mono text-xs">
      {/* Header */}
      <div className="p-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded bg-primary text-primary-foreground">
            <ShieldCheck className="h-3.5 w-3.5" />
          </div>
          <span className="font-bold text-foreground">Knowledge Base</span>
        </div>

        <div className="flex items-center gap-1">
          <ThemeToggle />
          <Button variant="ghost" size="icon-xs" onClick={onToggle} title="Collapse Sidebar">
            <ChevronLeft className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* New Session Button */}
      <div className="p-3 border-b border-border space-y-2">
        <Button
          variant="default"
          size="sm"
          onClick={onNewChat}
          className="w-full justify-start font-mono text-xs"
        >
          <Plus className="mr-1.5 h-3.5 w-3.5" /> New Query Session
        </Button>

        {/* Upload Button */}
        <Button
          variant="outline"
          size="sm"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
          className="w-full justify-start font-mono text-xs border-border"
        >
          <UploadCloud className="mr-1.5 h-3.5 w-3.5 text-primary" /> Upload PDF Document
        </Button>
        <input
          type="file"
          ref={fileInputRef}
          accept=".pdf,.txt"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFileUpload(file);
          }}
        />
      </div>

      {/* Status or Upload Message */}
      {uploading && (
        <div className="px-3 py-2 bg-primary/10 border-b border-primary/20 text-primary text-[11px] flex items-center gap-2">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          <span className="truncate">{uploadProgress}</span>
        </div>
      )}

      {error && (
        <div className="px-3 py-2 bg-destructive/10 border-b border-destructive/20 text-destructive text-[11px] flex items-center justify-between">
          <span className="truncate">{error}</span>
          <button onClick={() => setError(null)} className="font-bold">×</button>
        </div>
      )}

      {/* Settings & Scoping */}
      <div className="p-3 border-b border-border space-y-2 text-[11px]">
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground font-sans">NLI Self-Correction</span>
          <button
            type="button"
            onClick={onToggleCorrection}
            className={`relative inline-flex h-3.5 w-6 shrink-0 cursor-pointer rounded-full transition-colors ${
              enableCorrection ? "bg-primary" : "bg-muted-foreground/30"
            }`}
          >
            <span
              className={`pointer-events-none inline-block h-2.5 w-2.5 transform rounded-full bg-background transition ${
                enableCorrection ? "translate-x-2.5" : "translate-x-0"
              }`}
            />
          </button>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-muted-foreground font-sans">Subsystem Analytics</span>
          <Button
            variant={showAnalytics ? "secondary" : "ghost"}
            size="icon-xs"
            onClick={onToggleAnalytics}
          >
            <BarChart2 className="h-3.5 w-3.5 text-primary" />
          </Button>
        </div>
      </div>

      {/* Document List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        <div className="flex items-center justify-between text-muted-foreground text-[10px] uppercase font-bold tracking-wider mb-1">
          <span>Indexed Docs ({documents.length})</span>
          <button
            onClick={fetchDocuments}
            disabled={loadingDocs}
            className="hover:text-foreground"
            title="Refresh list"
          >
            <RefreshCw className={`h-3 w-3 ${loadingDocs ? "animate-spin text-primary" : ""}`} />
          </button>
        </div>

        {/* All Documents Scope Option */}
        <button
          onClick={() => onSelectDocument(undefined)}
          className={`w-full text-left p-2 rounded-md border text-xs transition-all flex items-center justify-between ${
            !selectedDocumentId
              ? "border-primary bg-primary/10 text-primary font-bold"
              : "border-border bg-background hover:bg-muted text-muted-foreground"
          }`}
        >
          <span className="truncate">✓ Target: All Indexed Docs</span>
        </button>

        {loadingDocs ? (
          <div className="flex items-center justify-center p-4 text-muted-foreground text-xs">
            <Loader2 className="h-4 w-4 animate-spin mr-2 text-primary" /> Loading index...
          </div>
        ) : documents.length === 0 ? (
          <div className="p-4 border border-dashed border-border rounded-lg text-center text-muted-foreground text-[11px]">
            <Layers className="h-5 w-5 mx-auto mb-1 opacity-50" />
            No documents uploaded yet.
          </div>
        ) : (
          documents.map((doc) => {
            const isSelected = selectedDocumentId === doc.document_id;
            const isProcessed = doc.status === "PROCESSED";
            return (
              <div
                key={doc.document_id}
                onClick={() => onSelectDocument(doc.document_id)}
                className={`group p-2 rounded-md border text-xs transition-all cursor-pointer space-y-1 ${
                  isSelected
                    ? "border-primary bg-primary/10 text-foreground font-bold"
                    : "border-border bg-background hover:border-primary/50 text-muted-foreground hover:text-foreground"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 truncate">
                    <FileText className="h-3.5 w-3.5 text-primary shrink-0" />
                    <span className="truncate">{doc.filename}</span>
                  </div>
                  <button
                    onClick={(e) => handleDelete(doc.document_id, e)}
                    className="opacity-0 group-hover:opacity-100 hover:text-destructive transition-opacity"
                    title="Delete"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>

                <div className="flex items-center justify-between text-[10px] text-muted-foreground font-sans">
                  <span>{doc.total_chunks || 0} chunks</span>
                  <span
                    className={`inline-flex items-center gap-0.5 px-1 py-0.2 rounded ${
                      isProcessed
                        ? "text-emerald-600 dark:text-emerald-400"
                        : "text-amber-600 dark:text-amber-400"
                    }`}
                  >
                    {isProcessed ? <CheckCircle className="h-2.5 w-2.5" /> : <Loader2 className="h-2.5 w-2.5 animate-spin" />}
                    {doc.status}
                  </span>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-border text-[10px] text-muted-foreground text-center font-sans">
        Self-Correcting RAG • DeBERTa-v3
      </div>
    </aside>
  );
}
