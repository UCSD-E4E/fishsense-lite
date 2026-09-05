import Link from "next/link";
import { redirect } from "next/navigation";
import { auth } from "@/auth";
import { isPortalAuthorized } from "@/lib/authz";
import { QUEUE_KINDS } from "@/lib/triage";
import { loadQueue } from "@/lib/triage-queue";
import { TriageClient } from "./triage-client";

export const dynamic = "force-dynamic";

export default async function TriagePage() {
  const session = await auth();
  if (!session?.user) {
    redirect(`/api/auth/signin?callbackUrl=${encodeURIComponent("/portal/triage")}`);
  }
  if (!isPortalAuthorized(session)) {
    redirect("/portal");
  }

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
    <main className="flex h-dvh flex-col bg-slate-950 text-slate-100">
      <header className="flex items-center justify-between gap-4 border-b border-slate-800 px-4 py-2">
        <div>
          <h1 className="text-sm font-semibold">{kind.label} triage</h1>
          <p className="text-xs text-slate-500">Accept the prediction, or skip it</p>
        </div>
        <nav className="text-xs">
          <Link href="/portal" className="text-slate-400 hover:text-slate-200">
            Portal
          </Link>
        </nav>
      </header>

      {error ? (
        <div className="p-6 text-sm text-amber-300">{error}</div>
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
