import Link from "next/link";
import { redirect } from "next/navigation";
import { auth } from "@/auth";
import { isPortalAuthorized } from "@/lib/authz";
import { QUEUE_KINDS, type QueueKind } from "@/lib/triage";
import { loadQueue } from "@/lib/triage-queue";
import { TriageClient } from "./triage-client";

export const dynamic = "force-dynamic";

function parseKind(value: string | undefined): QueueKind["key"] {
  return value === "headtail" ? "headtail" : "laser";
}

export default async function TriagePage({
  searchParams,
}: {
  searchParams: Promise<{ kind?: string }>;
}) {
  const session = await auth();
  if (!session?.user) {
    redirect(`/api/auth/signin?callbackUrl=${encodeURIComponent("/portal/triage")}`);
  }
  if (!isPortalAuthorized(session)) {
    redirect("/portal");
  }

  const kindKey = parseKind((await searchParams).kind);
  const kind = QUEUE_KINDS[kindKey];

  let items: Awaited<ReturnType<typeof loadQueue>>["items"] = [];
  let error: string | null = null;
  try {
    ({ items } = await loadQueue(kindKey));
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
        <nav className="flex items-center gap-2 text-xs">
          {Object.values(QUEUE_KINDS).map((k) => (
            <Link
              key={k.key}
              href={`/portal/triage?kind=${k.key}`}
              className={`rounded border px-2 py-1 ${
                k.key === kindKey
                  ? "border-sky-500 bg-sky-500/15 text-sky-300"
                  : "border-slate-700 text-slate-300 hover:bg-slate-800"
              }`}
            >
              {k.label}
            </Link>
          ))}
          <Link href="/portal" className="text-slate-400 hover:text-slate-200">
            Portal
          </Link>
        </nav>
      </header>

      {error ? (
        <div className="p-6 text-sm text-amber-300">{error}</div>
      ) : (
        <TriageClient items={items} kindLabel={kind.label} />
      )}
    </main>
  );
}
