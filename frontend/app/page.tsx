"use client";

import { useState } from "react";
import { Navbar } from "@/components/Navbar";
import { HeroSection } from "@/components/HeroSection";
import { FeaturesSection } from "@/components/FeaturesSection";
import { DocumentManager } from "@/components/DocumentManager";
import { QueryConsole } from "@/components/QueryConsole";
import { Footer } from "@/components/Footer";

export default function Home() {
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | undefined>(undefined);

  const scrollToSandbox = () => {
    const el = document.getElementById("sandbox");
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col selection:bg-indigo-500/30 selection:text-indigo-200">
      {/* Top Navbar */}
      <Navbar onNavigateToApp={scrollToSandbox} />

      {/* Main Page Layout */}
      <main className="flex-1">
        {/* Minimal Startup Hero Landing Section */}
        <HeroSection onLaunchSandbox={scrollToSandbox} />

        {/* 4 Subsystem Features & Architecture */}
        <FeaturesSection />

        {/* Document Ingestion & Knowledge Base Manager */}
        <DocumentManager
          selectedDocumentId={selectedDocumentId}
          onSelectDocument={(docId) => setSelectedDocumentId(docId)}
        />

        {/* Interactive Query Sandbox & Real-time SSE Stream Console */}
        <QueryConsole selectedDocumentId={selectedDocumentId} />
      </main>

      {/* Minimal Startup Footer */}
      <Footer />
    </div>
  );
}
