import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Groundwork | Document research",
  description: "Ask your documents questions and trace answers back to their sources.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
