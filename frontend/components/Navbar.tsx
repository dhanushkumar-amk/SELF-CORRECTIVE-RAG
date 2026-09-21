"use client";

import { useEffect, useState } from "react";
import { ShieldCheck, Sparkles, ArrowUpRight } from "lucide-react";
import { Button } from "@/components/ui/button";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface NavbarProps {
  onNavigateToApp?: () => void;
}

export function Navbar({ onNavigateToApp }: NavbarProps) {
  const [backendStatus, setBackendStatus] = useState<"connected" | "checking" | "disconnected">("checking");

  useEffect(() => {
    let active = true;
    const checkStatus = async () => {
      try {
        const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
        if (res.ok && active) {
          setBackendStatus("connected");
        } else if (active) {
          setBackendStatus("disconnected");
        }
      } catch {
        if (active) setBackendStatus("disconnected");
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/[0.08] bg-black/70 backdrop-blur-2xl transition-all">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4 sm:px-6">
        {/* Brand Logo */}
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-b from-white/15 to-white/5 border border-white/10 text-white shadow-inner">
            <ShieldCheck className="h-4 w-4 text-[#0a84ff]" />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold tracking-tight text-white font-sans">
              Self-Correcting RAG
            </span>
            <span className="rounded-full bg-white/[0.06] px-2 py-0.5 text-[10px] font-mono text-zinc-400 border border-white/[0.08]">
              DeBERTa-v3
            </span>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="hidden md:flex items-center gap-7 text-xs font-medium text-zinc-400">
          <a href="#overview" className="transition-colors hover:text-white">
            Overview
          </a>
          <a href="#features" className="transition-colors hover:text-white">
            Architecture
          </a>
          <a href="#documents" className="transition-colors hover:text-white">
            Knowledge Base
          </a>
          <a href="#sandbox" className="transition-colors hover:text-white">
            Sandbox
          </a>
          <a
            href={`${API_URL}/docs`}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1 text-[#0a84ff] hover:text-[#409cff] font-mono"
          >
            API <ArrowUpRight className="h-3 w-3" />
          </a>
        </nav>

        {/* Action Controls & Health Pill */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 rounded-full border border-white/[0.08] bg-white/[0.03] px-2.5 py-1 text-[11px] font-mono">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                backendStatus === "connected"
                  ? "bg-[#30d158] shadow-[0_0_8px_rgba(48,209,88,0.6)]"
                  : backendStatus === "checking"
                  ? "bg-[#ffd60a] animate-pulse"
                  : "bg-[#ff453a]"
              }`}
            />
            <span className="text-zinc-500 hidden sm:inline">Backend:</span>
            <span
              className={
                backendStatus === "connected"
                  ? "text-[#30d158]"
                  : backendStatus === "checking"
                  ? "text-[#ffd60a]"
                  : "text-[#ff453a]"
              }
            >
              {backendStatus}
            </span>
          </div>

          <Button
            size="sm"
            onClick={onNavigateToApp}
            className="bg-[#0071e3] hover:bg-[#0077ed] active:scale-[0.98] text-white font-medium text-xs rounded-full px-4 h-8 shadow-sm transition-all"
          >
            <Sparkles className="mr-1.5 h-3.5 w-3.5 text-white/80" /> Playground
          </Button>
        </div>
      </div>
    </header>
  );
}
