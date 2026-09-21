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
        setError("Failed to load documents from backend");
      }
    } catch (err: any) {
      setError(err?.message || "Backend connection error");
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
    if (!confirm("Remove document from knowledge index?")) return;

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
    <section id="documents" className="py-16 border-t border-white/[0.08]">
      <div className="mx-auto max-w-5xl px-4 sm:px-6">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              <FileText className="h-5 w-5 text-[#0a84ff]" /> Knowledge Base Documents
            </h2>
            <p className="text-xs text-zinc-400 mt-0.5">
              Upload PDF documents to parse, chunk, and index into FAISS + BM25 vector memory.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={fetchDocuments}
              disabled={loadingDocs}
              className="border-white/10 bg-white/[0.04] text-xs text-zinc-300 hover:bg-white/[0.08] hover:text-white rounded-full px-3.5"
            >
              <RefreshCw className={`mr-1.5 h-3.5 w-3.5 ${loadingDocs ? "animate-spin text-[#0a84ff]" : ""}`} />
              Sync Index
            </Button>
            <Button
              size="sm"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="bg-[#0071e3] hover:bg-[#0077ed] text-white font-medium text-xs rounded-full px-4 shadow-sm"
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

        {/* Upload Status Banner */}
        {uploading && (
          <div className="mb-6 rounded-2xl border border-[#0a84ff]/30 bg-[#0a84ff]/10 p-4 flex items-center justify-between text-xs text-[#0a84ff]">
            <div className="flex items-center gap-3">
              <Loader2 className="h-4 w-4 animate-spin text-[#0a84ff]" />
              <span>{uploadProgress || "Uploading and processing..."}</span>
            </div>
            <span className="font-mono text-[11px] text-[#0a84ff]/80">Processing</span>
          </div>
        )}

        {/* Error Alert */}
        {error && (
          <div className="mb-6 rounded-2xl border border-[#ff453a]/30 bg-[#ff453a]/10 p-4 flex items-center justify-between text-xs text-[#ff453a]">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-[#ff453a]" />
              <span>{error}</span>
            </div>
            <button onClick={() => setError(null)} className="text-[#ff453a] hover:text-white">
              Dismiss
            </button>
          </div>
        )}

        {/* Dropzone & Document List */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          {/* Upload Dropzone Box */}
          <div
            onClick={() => fileInputRef.current?.click()}
            className="group relative flex flex-col items-center justify-center rounded-2xl border border-dashed border-white/15 bg-white/[0.02] p-6 text-center cursor-pointer transition-all hover:border-[#0a84ff]/50 hover:bg-white/[0.04]"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white/[0.05] border border-white/10 mb-3 group-hover:scale-105 transition-transform">
              <UploadCloud className="h-5 w-5 text-[#0a84ff]" />
            </div>
            <p className="text-xs font-semibold text-white">Click or drag & drop PDF</p>
            <p className="text-[11px] text-zinc-500 mt-1">Up to 20MB document size</p>
            <span className="mt-3 rounded-full bg-white/[0.04] px-2.5 py-0.5 text-[10px] font-mono text-zinc-400 border border-white/[0.08]">
              FAISS Vector Indexing
            </span>
          </div>

          {/* Document Cards */}
          <div className="lg:col-span-2 space-y-2">
            <div className="flex items-center justify-between text-xs text-zinc-400 px-1">
              <span>Indexed Documents ({documents.length})</span>
              <button
                onClick={() => onSelectDocument(undefined)}
                className={`text-[11px] font-mono transition-colors ${
                  !selectedDocumentId ? "text-[#0a84ff] font-semibold" : "text-zinc-500 hover:text-white"
                }`}
              >
                {!selectedDocumentId ? "✓ Target: All Documents" : "Target: All Documents"}
              </button>
            </div>

            {loadingDocs ? (
              <div className="flex items-center justify-center p-8 rounded-2xl border border-white/[0.08] bg-white/[0.02]">
                <Loader2 className="h-5 w-5 animate-spin text-[#0a84ff]" />
                <span className="ml-2 text-xs text-zinc-400">Loading document index...</span>
              </div>
            ) : documents.length === 0 ? (
              <div className="flex flex-col items-center justify-center p-8 rounded-2xl border border-white/[0.08] bg-white/[0.02] text-center">
                <Layers className="h-7 w-7 text-zinc-600 mb-2" />
                <p className="text-xs font-medium text-zinc-400">No documents indexed yet.</p>
                <p className="text-[11px] text-zinc-500 mt-0.5">
                  Upload a document above to test hallucination detection on your custom files.
                </p>
              </div>
            ) : (
              <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1">
                {documents.map((doc) => {
                  const isSelected = selectedDocumentId === doc.document_id;
                  const isProcessed = doc.status === "PROCESSED";
                  return (
                    <div
                      key={doc.document_id}
                      onClick={() => onSelectDocument(doc.document_id)}
                      className={`flex items-center justify-between p-3.5 rounded-xl border transition-all cursor-pointer ${
                        isSelected
                          ? "border-[#0a84ff]/60 bg-[#0a84ff]/10"
                          : "border-white/[0.08] bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]"
                      }`}
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <div className={`p-2 rounded-lg ${isSelected ? "bg-[#0a84ff]/20 text-[#0a84ff]" : "bg-white/[0.05] text-zinc-400"}`}>
                          <FileText className="h-4 w-4" />
                        </div>
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <h4 className="text-xs font-semibold text-white truncate">{doc.filename}</h4>
                            {isSelected && (
                              <span className="rounded-full bg-[#0a84ff]/20 px-2 py-0.5 text-[9px] font-mono text-[#0a84ff] border border-[#0a84ff]/30">
                                Target
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-zinc-500 font-mono mt-0.5">
                            <span>ID: {doc.document_id.slice(0, 8)}...</span>
                            {doc.total_chunks !== undefined && (
                              <span>• {doc.total_chunks} chunks</span>
                            )}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <span
                          className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[9px] font-medium border ${
                            isProcessed
                              ? "bg-[#30d158]/10 text-[#30d158] border-[#30d158]/20"
                              : doc.status === "PROCESSING"
                              ? "bg-[#ffd60a]/10 text-[#ffd60a] border-[#ffd60a]/20"
                              : "bg-[#ff453a]/10 text-[#ff453a] border-[#ff453a]/20"
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

                        <button
                          onClick={(e) => handleDelete(doc.document_id, e)}
                          className="text-zinc-500 hover:text-[#ff453a] p-1 rounded transition-colors"
                          title="Delete Document"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
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
