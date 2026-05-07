import type { Metadata } from "next";
import pkg from "../package.json";
import "./globals.css";

export const metadata: Metadata = {
  title: "E4E FishSense",
  description: "FishSense dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="flex min-h-screen flex-col bg-slate-50 text-slate-900 antialiased dark:bg-slate-950 dark:text-slate-100">
        <div className="flex-1">{children}</div>
        <footer className="border-t border-slate-200 px-6 py-3 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
          fishsense-lite-web v{pkg.version}
        </footer>
      </body>
    </html>
  );
}
