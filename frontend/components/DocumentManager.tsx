"use client";

import { useEffect, useState, useRef } from "react";
import {
  UploadCloud,
  FileText,
  CheckCircle,
  AlertCircle,
  Loader2,
  Trash2,
  RefreshCw,
  Layers,
} from "lucide-react";
import { Button } from "@/components/ui/button";

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

interface DocumentManagerProps {
  selectedDocumentId?: string;
  onSelectDocument: (docId?: string) => void;
}

export function DocumentManager({ selectedDocumentId, onSelectDocument }: DocumentManagerProps) {
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
        setError("Failed to fetch documents from server");
      }
    } catch (err: any) {
      setError(err?.message || "Could not connect to backend server");
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
        throw new Error(errJson.detail || errJson.message || "Failed to upload document");
      }

      const uploadData = await uploadRes.json();
      const docId = uploadData.document_id || uploadData.id;

      setUploadProgress(`Processing & indexing chunks for ${file.name}...`);
      const processRes = await fetch(`${API_URL}/api/v1/documents/${docId}/process`, {
        method: "POST",
      });

      if (!processRes.ok) {
        const errJson = await processRes.json().catch(() => ({}));
        throw new Error(errJson.detail || errJson.message || "Failed to process document");
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
    if (!confirm("Delete document from knowledge index?")) return;

    try {
      const res = await fetch(`${API_URL}/api/v1/documents/${docId}`, {
        method: "DELETE",
      });
      if (res.ok) {
        if (selectedDocumentId === docId) {
          onSelectDocument(undefined);
        }
        await fetchDocuments();
      } else {
        alert("Failed to delete document");
      }
    } catch {
      alert("Error deleting document");
    }
  };

  return (
    <section id="documents" className="py-14 border-b border-border">
      <div className="mx-auto max-w-5xl px-4 sm:px-6">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-xl font-bold tracking-tight text-foreground font-mono flex items-center gap-2">
              <FileText className="h-5 w-5 text-primary" /> Knowledge Base Index
            </h2>
            <p className="text-xs text-muted-foreground mt-0.5 font-sans">
              Upload PDF documents to parse, chunk, and index into FAISS + BM25 vector memory.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={fetchDocuments}
              disabled={loadingDocs}
              className="text-xs font-mono border-border"
            >
              <RefreshCw className={`mr-1.5 h-3.5 w-3.5 ${loadingDocs ? "animate-spin text-primary" : ""}`} />
              Refresh
            </Button>
            <Button
              variant="default"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="text-xs font-mono"
            >
              <UploadCloud className="mr-1.5 h-3.5 w-3.5" />
              Upload PDF
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
        </div>

        {/* Upload Banner */}
        {uploading && (
          <div className="mb-4 rounded-lg border border-primary/40 bg-primary/10 p-3 flex items-center justify-between text-xs text-foreground font-mono">
            <div className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin text-primary" />
              <span>{uploadProgress || "Uploading and processing..."}</span>
            </div>
            <span>Indexing</span>
          </div>
        )}

        {/* Error Alert */}
        {error && (
          <div className="mb-4 rounded-lg border border-destructive/40 bg-destructive/10 p-3 flex items-center justify-between text-xs text-destructive font-mono">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4" />
              <span>{error}</span>
            </div>
            <Button variant="ghost" size="xs" onClick={() => setError(null)}>
              Dismiss
            </Button>
          </div>
        )}

        {/* Dropzone & Document Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div
            onClick={() => fileInputRef.current?.click()}
            className="group relative flex flex-col items-center justify-center rounded-xl border border-dashed border-border bg-card p-6 text-center cursor-pointer transition-all hover:border-primary hover:bg-muted/40 shadow-sm"
          >
            <div className="flex h-9 w-9 items-center justify-center rounded-md bg-muted text-primary mb-2 group-hover:scale-105 transition-transform border border-border">
              <UploadCloud className="h-4.5 w-4.5" />
            </div>
            <p className="text-xs font-bold text-foreground font-mono">Click or Drag PDF</p>
            <p className="text-[11px] text-muted-foreground mt-0.5 font-sans">PDF & Text up to 20MB</p>
            <span className="mt-3 rounded bg-muted px-2 py-0.5 text-[10px] font-mono text-muted-foreground border border-border">
              FAISS Indexing
            </span>
          </div>

          <div className="lg:col-span-2 space-y-2">
            <div className="flex items-center justify-between text-xs text-muted-foreground font-mono px-1">
              <span>Documents ({documents.length})</span>
              <button
                onClick={() => onSelectDocument(undefined)}
                className={`text-[11px] font-mono transition-colors ${
                  !selectedDocumentId ? "text-primary font-bold" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {!selectedDocumentId ? "✓ Scoped to All Documents" : "Target: All Documents"}
              </button>
            </div>

            {loadingDocs ? (
              <div className="flex items-center justify-center p-6 rounded-xl border border-border bg-card">
                <Loader2 className="h-4 w-4 animate-spin text-primary" />
                <span className="ml-2 text-xs text-muted-foreground font-mono">Loading document index...</span>
              </div>
            ) : documents.length === 0 ? (
              <div className="flex flex-col items-center justify-center p-6 rounded-xl border border-border bg-card text-center">
                <Layers className="h-6 w-6 text-muted-foreground mb-2" />
                <p className="text-xs font-semibold text-muted-foreground font-mono">No documents indexed yet.</p>
              </div>
            ) : (
              <div className="space-y-2 max-h-[260px] overflow-y-auto pr-1">
                {documents.map((doc) => {
                  const isSelected = selectedDocumentId === doc.document_id;
                  const isProcessed = doc.status === "PROCESSED";
                  return (
                    <div
                      key={doc.document_id}
                      onClick={() => onSelectDocument(doc.document_id)}
                      className={`flex items-center justify-between p-3 rounded-lg border transition-all cursor-pointer ${
                        isSelected
                          ? "border-primary bg-primary/10 shadow-sm"
                          : "border-border bg-card hover:border-primary/50"
                      }`}
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <div className={`p-1.5 rounded ${isSelected ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"}`}>
                          <FileText className="h-4 w-4" />
                        </div>
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <h4 className="text-xs font-bold text-foreground font-mono truncate">{doc.filename}</h4>
                            {isSelected && (
                              <span className="rounded bg-primary/20 px-1.5 py-0.5 text-[9px] font-mono text-primary border border-primary/30">
                                Target
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-muted-foreground font-mono mt-0.5">
                            <span>ID: {doc.document_id.slice(0, 8)}...</span>
                            {doc.total_chunks !== undefined && <span>• {doc.total_chunks} chunks</span>}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <span
                          className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-[9px] font-mono border ${
                            isProcessed
                              ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
                              : doc.status === "PROCESSING"
                              ? "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20"
                              : "bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20"
                          }`}
                        >
                          {isProcessed ? (
                            <CheckCircle className="h-2.5 w-2.5" />
                          ) : doc.status === "PROCESSING" ? (
                            <Loader2 className="h-2.5 w-2.5 animate-spin" />
                          ) : (
                            <AlertCircle className="h-2.5 w-2.5" />
                          )}
                          {doc.status}
                        </span>

                        <Button
                          variant="ghost"
                          size="icon-xs"
                          onClick={(e) => handleDelete(doc.document_id, e)}
                          title="Delete Document"
                        >
                          <Trash2 className="h-3.5 w-3.5 text-muted-foreground hover:text-destructive" />
                        </Button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
