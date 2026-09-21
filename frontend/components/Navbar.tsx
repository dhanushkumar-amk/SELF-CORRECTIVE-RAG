"use client";

import { useEffect, useState } from "react";
import { ShieldCheck, Sparkles, ArrowUpRight, Cpu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/ThemeToggle";

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
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/90 backdrop-blur-md transition-colors">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4 sm:px-6">
        {/* Brand Logo MCP Style */}
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-sm">
            <ShieldCheck className="h-4 w-4" />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold tracking-tight text-foreground font-mono">
              Self-Correcting RAG
            </span>
            <span className="rounded bg-muted px-2 py-0.5 text-[10px] font-mono text-muted-foreground border border-border">
              DeBERTa NLI
            </span>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="hidden md:flex items-center gap-6 text-xs font-medium text-muted-foreground font-mono">
          <a href="#overview" className="transition-colors hover:text-foreground">
            Overview
          </a>
          <a href="#features" className="transition-colors hover:text-foreground">
            Architecture
          </a>
          <a href="#documents" className="transition-colors hover:text-foreground">
            Knowledge Base
          </a>
          <a href="#sandbox" className="transition-colors hover:text-foreground">
            Sandbox
          </a>
          <a
            href={`${API_URL}/docs`}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1 text-primary hover:underline"
          >
            API Docs <ArrowUpRight className="h-3 w-3" />
          </a>
        </nav>

        {/* Action Controls, Health Pill & Theme Toggle */}
        <div className="flex items-center gap-2.5">
          <div className="flex items-center gap-2 rounded-md border border-border bg-muted/50 px-2.5 py-1 text-[11px] font-mono">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                backendStatus === "connected"
                  ? "bg-emerald-500"
                  : backendStatus === "checking"
                  ? "bg-amber-500 animate-pulse"
                  : "bg-rose-500"
              }`}
            />
            <span className="text-muted-foreground hidden sm:inline">Backend:</span>
            <span
              className={
                backendStatus === "connected"
                  ? "text-emerald-600 dark:text-emerald-400 font-semibold"
                  : backendStatus === "checking"
                  ? "text-amber-600 dark:text-amber-400 font-semibold"
                  : "text-rose-600 dark:text-rose-400 font-semibold"
              }
            >
              {backendStatus}
            </span>
          </div>

          {/* Dark / Light Mode Toggle Button */}
          <ThemeToggle />

          {/* Launch Sandbox Button using shadcn Button */}
          <Button
            variant="default"
            size="sm"
            onClick={onNavigateToApp}
            className="font-medium text-xs font-mono"
          >
            <Sparkles className="mr-1.5 h-3.5 w-3.5" /> Launch Sandbox
          </Button>
        </div>
      </div>
    </header>
  );
}
