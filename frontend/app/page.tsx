"use client";

import { useEffect, useState } from "react";

type BackendStatus = "checking" | "connected" | "disconnected";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [status, setStatus] = useState<BackendStatus>("checking");

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
        if (res.ok) {
          const data = await res.json();
          setStatus(data.status === "ok" ? "connected" : "disconnected");
        } else {
          setStatus("disconnected");
        }
      } catch {
        setStatus("disconnected");
      }
    };

    checkHealth();
  }, []);

  const statusColor: Record<BackendStatus, string> = {
    checking: "text-yellow-500",
    connected: "text-emerald-500",
    disconnected: "text-red-500",
  };

  const statusDot: Record<BackendStatus, string> = {
    checking: "bg-yellow-500 animate-pulse",
    connected: "bg-emerald-500",
    disconnected: "bg-red-500",
  };

  return (
    <main className="flex flex-1 flex-col items-center justify-center gap-8 p-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
          Self-Correcting RAG
        </h1>
        <p className="text-lg text-foreground/60">
          with Hallucination Detection
        </p>
      </div>

      <div className="flex items-center gap-3 rounded-full border border-foreground/10 bg-foreground/5 px-6 py-3">
        <span
          className={`inline-block h-3 w-3 rounded-full ${statusDot[status]}`}
        />
        <span className={`font-mono text-sm ${statusColor[status]}`}>
          Backend:{" "}
          {status === "checking" ? "checking…" : status}
        </span>
      </div>

      <p className="max-w-md text-center text-sm text-foreground/40">
        Phase 1 — Project scaffolding complete. The system will be built
        incrementally across 50 phases.
      </p>
    </main>
  );
}
