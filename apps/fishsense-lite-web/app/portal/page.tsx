import { auth, signOut } from "@/auth";

export const dynamic = "force-dynamic";

export default async function PortalPage() {
  const session = await auth();
  const user = session?.user;

  return (
    <main className="mx-auto max-w-4xl px-6 py-12">
      <header className="mb-8 flex items-center justify-between">
        <h1 className="text-3xl font-semibold tracking-tight">Portal</h1>
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
      </header>

      <section className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <h2 className="text-lg font-medium">Signed in as</h2>
        <dl className="mt-3 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-2 text-sm">
          <dt className="text-slate-500">Name</dt>
          <dd>{user?.name ?? "—"}</dd>
          <dt className="text-slate-500">Email</dt>
          <dd>{user?.email ?? "—"}</dd>
          <dt className="text-slate-500">Groups</dt>
          <dd>{user?.groups?.length ? user.groups.join(", ") : "—"}</dd>
        </dl>
      </section>
    </main>
  );
}
