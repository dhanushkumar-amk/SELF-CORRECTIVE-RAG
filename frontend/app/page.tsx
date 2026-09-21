"use client";

import { useState } from "react";
import { ThemeProvider } from "@/components/ThemeProvider";
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
    <ThemeProvider>
      <div className="min-h-screen bg-background text-foreground flex flex-col selection:bg-primary selection:text-primary-foreground transition-colors duration-200">
        {/* Top Navbar */}
        <Navbar onNavigateToApp={scrollToSandbox} />

        {/* Main Content */}
        <main className="flex-1">
          {/* Hero Section */}
          <HeroSection onLaunchSandbox={scrollToSandbox} />

          {/* Features Grid */}
          <FeaturesSection />

          {/* Knowledge Base & Document Management */}
          <DocumentManager
            selectedDocumentId={selectedDocumentId}
            onSelectDocument={(docId) => setSelectedDocumentId(docId)}
          />

          {/* Interactive Console & Analytics Charts */}
          <QueryConsole selectedDocumentId={selectedDocumentId} />
        </main>

        {/* Footer */}
        <Footer />
      </div>
    </ThemeProvider>
  );
}
