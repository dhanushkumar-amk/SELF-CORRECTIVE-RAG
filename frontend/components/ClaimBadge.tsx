"use client";

/**
 * Per-claim verification badge (Phase 48).
 *
 * Maps get_claim_final_status() categories to visual treatments and opens a
 * popover with the exact source passage, page, and document the claim is
 * grounded in. Claims that went through the correction loop and came out
 * verified carry a distinct "Corrected after review" treatment (Task 10).
 */

import {
  Check,
  AlertTriangle,
  X,
  HelpCircle,
  RefreshCw,
  FileText,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import {
  getClaimStatus,
  type ClaimStatus,
  type ClaimWithSource,
} from "@/lib/api";

const STATUS_STYLES: Record<
  ClaimStatus,
  { label: string; className: string; icon: React.ReactNode }
> = {
  verified: {
    label: "Verified",
    className: "bg-emerald-500/10 text-emerald-600 border-emerald-500/20 dark:text-emerald-400",
    icon: <Check className="h-3 w-3" />,
  },
  needs_review: {
    label: "Needs Review",
    className: "bg-amber-500/10 text-amber-600 border-amber-500/20 dark:text-amber-400",
    icon: <AlertTriangle className="h-3 w-3" />,
  },
  contradicted: {
    label: "Contradicted",
    className: "bg-red-500/10 text-red-600 border-red-500/20 dark:text-red-400",
    icon: <X className="h-3 w-3" />,
  },
  unverifiable: {
    label: "Unverifiable",
    className: "bg-zinc-500/10 text-zinc-500 border-zinc-500/20 dark:text-zinc-400",
    icon: <HelpCircle className="h-3 w-3" />,
  },
  pending: {
    label: "Pending",
    className: "bg-muted text-muted-foreground border-border",
    icon: <HelpCircle className="h-3 w-3" />,
  },
};

function StatusBadge({ status }: { status: ClaimStatus }) {
  const style = STATUS_STYLES[status];
  return (
    <span
      className={cn(
        "inline-flex h-5 shrink-0 items-center gap-1 rounded-full border px-2 text-[11px] font-medium",
        style.className,
      )}
    >
      {style.icon}
      {style.label}
    </span>
  );
}

export function ClaimBadge({ claim }: { claim: ClaimWithSource }) {
  const status = getClaimStatus(claim);
  const correctedAfterReview = claim.was_corrected === true && status === "verified";
  const correctionUnresolved = claim.was_corrected === true && status !== "verified";
  const confidencePercent =
    claim.confidence != null ? `${(claim.confidence * 100).toFixed(0)}%` : null;

  return (
    <div className="flex items-center gap-2">
      <Popover>
        <PopoverTrigger
          className="rounded-full outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
          aria-label={`Claim status ${status}. Click for source passage.`}
        >
          <StatusBadge status={status} />
        </PopoverTrigger>

        <PopoverContent
          align="start"
          sideOffset={6}
          className="w-80 max-w-[calc(100vw-2rem)]"
        >
          <div className="flex items-center justify-between gap-2">
            <span className="flex items-center gap-1.5 text-[11px] font-semibold text-foreground">
              <FileText className="h-3 w-3 text-muted-foreground" />
              Source Passage
            </span>
            {confidencePercent && (
              <span className="font-mono text-[11px] text-muted-foreground">
                NLI {confidencePercent}
              </span>
            )}
          </div>

          {claim.filename && (
            <p className="text-[11px] text-muted-foreground">
              {claim.filename}
              {claim.page_number != null &&
                ` · p. ${claim.page_number}${claim.page_number_end && claim.page_number_end !== claim.page_number ? `–${claim.page_number_end}` : ""}`}
            </p>
          )}

          <div className="max-h-36 overflow-y-auto rounded-md border border-border bg-muted/50 p-2.5 text-[12px] leading-relaxed text-foreground/85">
            {claim.source_text || "Source passage text is stored in the vector index."}
          </div>

          {correctedAfterReview && (
            <div className="flex items-center gap-1.5 rounded-md border border-emerald-500/20 bg-emerald-500/10 px-2.5 py-1.5 text-[11px] font-medium text-emerald-600 dark:text-emerald-400">
              <RefreshCw className="h-3 w-3" />
              Corrected after review — this claim was flagged by NLI, re-grounded
              against new context, and passed on re-verification.
            </div>
          )}
          {correctionUnresolved && (
            <div className="flex items-center gap-1.5 rounded-md border border-red-500/20 bg-red-500/10 px-2.5 py-1.5 text-[11px] font-medium text-red-600 dark:text-red-400">
              <RefreshCw className="h-3 w-3" />
              Correction attempted, but this claim still failed verification
              after the retry loop.
            </div>
          )}
          {!claim.was_corrected && status !== "verified" && status !== "pending" && (
            <p className="text-[11px] text-muted-foreground">
              {status === "unverifiable"
                ? "No matching source chunk was found for this claim."
                : status === "needs_review"
                  ? "The evidence is borderline or insufficient to confirm this claim."
                  : "The source passage contradicts this claim."}
            </p>
          )}
        </PopoverContent>
      </Popover>

      {correctedAfterReview && (
        <span className="hidden items-center gap-1 text-[11px] font-medium text-emerald-600 dark:text-emerald-400 sm:inline-flex">
          <RefreshCw className="h-3 w-3" />
          Corrected after review
        </span>
      )}
    </div>
  );
}

/** Badge-only variant (no popover) for compact lists. */
export function ClaimStatusBadge({ claim }: { claim: ClaimWithSource }) {
  return <StatusBadge status={getClaimStatus(claim)} />;
}

export { Badge };
