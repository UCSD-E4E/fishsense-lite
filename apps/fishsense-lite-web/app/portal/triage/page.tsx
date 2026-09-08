import Link from "next/link";
import { QUEUE_KINDS } from "@/lib/triage";
import { loadQueue } from "@/lib/triage-queue";
import { requirePortalUser } from "../guard";
import { TriageClient } from "./triage-client";

export const dynamic = "force-dynamic";

export default async function TriagePage() {
  await requirePortalUser("/portal/triage");

  const kind = QUEUE_KINDS.laser;

  let queue: Awaited<ReturnType<typeof loadQueue>> = {
    items: [],
    scanned: 0,
    projects: [],
    notWalked: 0,
  };
  let error: string | null = null;
  try {
    queue = await loadQueue(kind.key);
  } catch (e) {
    error = e instanceof Error ? e.message : "Could not load the queue";
  }

  return (
    // Inherits the root layout's background and theme like every other route.
    // It used to hardcode `bg-slate-950 text-slate-100` and `h-dvh`, which
    // painted over the layout's chrome and stayed dark in light mode — so the
    // page read as a different application.
    <main className="mx-auto flex w-full max-w-5xl flex-col gap-4 px-4 py-6">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            {kind.label} triage
          </h1>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
            Accept the prediction, or skip it.
          </p>
        </div>
        <Link
          href="/portal"
          className="rounded-md border border-slate-300 bg-white px-3 py-1 text-sm font-medium shadow-sm transition hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:hover:bg-slate-800"
        >
          Portal
        </Link>
      </header>

      {error ? (
        <p className="rounded-md border border-amber-400 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-700/60 dark:bg-amber-950/40 dark:text-amber-200">
          {error}
        </p>
      ) : (
        <TriageClient
          items={queue.items}
          kindLabel={kind.label}
          scanned={queue.scanned}
          projects={queue.projects}
          notWalked={queue.notWalked}
        />
      )}
    </main>
  );
}
