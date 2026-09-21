"use client";

import { Sun, Moon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/components/ThemeProvider";

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <Button
      variant="outline"
      size="icon-sm"
      onClick={toggleTheme}
      title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      className="border-border bg-background hover:bg-muted text-foreground font-mono"
    >
      {theme === "dark" ? (
        <Sun className="h-4 w-4 text-amber-400 transition-transform" />
      ) : (
        <Moon className="h-4 w-4 text-slate-700 transition-transform" />
      )}
      <span className="sr-only">Toggle theme</span>
    </Button>
  );
}
