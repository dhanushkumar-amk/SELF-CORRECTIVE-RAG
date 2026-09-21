/**
 * Utility helper for conditional class merging.
 */
export function cn(...inputs: (string | boolean | undefined | null | {[key: string]: boolean})[]): string {
  const classes: string[] = [];

  for (const input of inputs) {
    if (!input) continue;
    if (typeof input === "string") {
      classes.push(input.trim());
    } else if (typeof input === "object") {
      for (const [key, val] of Object.entries(input)) {
        if (val) classes.push(key.trim());
      }
    }
  }

  return classes.filter(Boolean).join(" ");
}
