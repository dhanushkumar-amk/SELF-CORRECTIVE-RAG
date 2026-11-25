"use client";

import { Dialog as DialogPrimitive } from "@base-ui/react/dialog";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const Dialog = DialogPrimitive.Root;
const DialogTitle = DialogPrimitive.Title;
const DialogDescription = DialogPrimitive.Description;

function DialogContent({ className, children, showClose = true, placement = "center", ...props }:
  Omit<DialogPrimitive.Popup.Props, "className"> & { className?: string; showClose?: boolean; placement?: "center" | "left" }) {
  return (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Backdrop className="fixed inset-0 z-40 bg-[#100c24]/55 backdrop-blur-xs" />
      <DialogPrimitive.Popup
        className={cn(
          "fixed z-50 overflow-y-auto bg-card text-card-foreground shadow-xl outline-none",
          placement === "left"
            ? "inset-y-0 left-0 w-[min(20rem,90vw)]"
            : "top-1/2 left-1/2 max-h-[90dvh] w-[calc(100%-2rem)] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-2xl border border-border p-6",
          className,
        )}
        {...props}
      >
        {children}
        {showClose && (
          <DialogPrimitive.Close
            render={<Button variant="ghost" size="icon" className="absolute top-3 right-3" />}
            aria-label="Close dialog"
          >
            <X />
          </DialogPrimitive.Close>
        )}
      </DialogPrimitive.Popup>
    </DialogPrimitive.Portal>
  );
}

export { Dialog, DialogContent, DialogDescription, DialogTitle };
