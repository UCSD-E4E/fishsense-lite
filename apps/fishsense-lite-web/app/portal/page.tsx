import { redirect } from "next/navigation";
import { auth, signOut } from "@/auth";
import { getDives } from "@/lib/dives";
import { CalibrationLinks } from "./calibration-links";

export const dynamic = "force-dynamic";

export default async function PortalPage() {
  const session = await auth();
  if (!session?.user) {
    redirect(`/api/auth/signin?callbackUrl=${encodeURIComponent("/portal")}`);
  }
  const user = session.user;

  let dives: Awaited<ReturnType<typeof getDives>> = [];
  let divesError: string | null = null;
  try {
    dives = await getDives();
  } catch (error) {
    divesError = error instanceof Error ? error.message : "Failed to load dives";
  }

  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <header className="mb-8 flex items-start justify-between gap-4">
        <h1 className="text-3xl font-semibold tracking-tight">Portal</h1>
        <div className="flex items-start gap-4">
          <div className="text-right text-sm leading-tight">
            <div className="font-medium">{user?.name ?? "—"}</div>
            <div className="text-slate-500">{user?.email ?? "—"}</div>
            <div className="text-xs text-slate-400">
              {user?.groups?.length ? user.groups.join(", ") : "no groups"}
            </div>
          </div>
          <form
            action={async () => {
              "use server";
              await signOut({ redirectTo: "/" });
            }}
          >
            <button
              type="submit"
              className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium shadow-sm transition hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:hover:bg-slate-800"
            >
              Sign out
            </button>
          </form>
        </div>
      </header>

      <section className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <h2 className="text-lg font-medium">Dive calibration links</h2>
        <p className="mt-1 max-w-3xl text-sm text-slate-600 dark:text-slate-400">
          Link a dive that has no slate of its own to a sibling slate dive shot
          with the same camera and laser rig. The dive then borrows that dive&apos;s
          laser calibration, so it can be measured without a slate in-frame. A
          dive with its own slate self-calibrates and needs no link.
        </p>

        <div className="mt-6">
          {divesError ? (
            <p className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:border-red-700 dark:bg-red-950 dark:text-red-300">
              Could not load dives: {divesError}
            </p>
          ) : (
            <CalibrationLinks dives={dives} />
          )}
        </div>
      </section>
    </main>
  );
}
