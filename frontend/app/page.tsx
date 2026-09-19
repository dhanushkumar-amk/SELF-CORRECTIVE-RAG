"use client";

import { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";

type BackendStatus = "checking" | "connected" | "disconnected";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [status, setStatus] = useState<BackendStatus>("checking");

  const checkHealth = useCallback(async () => {
    setStatus("checking");
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
  }, []);

  useEffect(() => {
    let isCancelled = false;
    fetch(`${API_URL}/health`, { cache: "no-store" })
      .then(async (res) => {
        if (!res.ok) throw new Error("Health check failed");
        return res.json();
      })
      .then((data) => {
        if (!isCancelled) {
          setStatus(data.status === "ok" ? "connected" : "disconnected");
        }
      })
      .catch(() => {
        if (!isCancelled) {
          setStatus("disconnected");
        }
      });

    return () => {
      isCancelled = true;
    };
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

      <div className="flex flex-col items-center gap-4">
        <div className="flex items-center gap-3 rounded-full border border-foreground/10 bg-foreground/5 px-6 py-3">
          <span
            className={`inline-block h-3 w-3 rounded-full ${statusDot[status]}`}
          />
          <span className={`font-mono text-sm ${statusColor[status]}`}>
            Backend:{" "}
            {status === "checking" ? "checking…" : status}
          </span>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={checkHealth}
          disabled={status === "checking"}
        >
          Re-check Connection
        </Button>
      </div>

      <p className="max-w-md text-center text-sm text-foreground/40">
        Phase 2 — Environment setup complete. The system is configured for
        reproducible development across backend and frontend.
      </p>
    </main>
  );
}
