import { getActiveProjects } from "@/lib/active-projects";
import { buildSections, type Section, type SectionLink } from "@/lib/sections";
import { ADMIN_LINKS, RESULTS_LINKS } from "@/lib/static-links";

export const dynamic = "force-dynamic";

const FETCH_REVALIDATE_SECONDS = 300;

export default async function HomePage() {
  const active = await getActiveProjects(FETCH_REVALIDATE_SECONDS);
  const sections = buildSections(active, {
    results: RESULTS_LINKS,
    admin: ADMIN_LINKS,
  });

  return (
    <main className="mx-auto max-w-6xl px-6 py-12">
      <header className="mb-10 flex items-center justify-between gap-4">
        <h1 className="text-4xl font-semibold tracking-tight">E4E FishSense</h1>
        <a
          href="/portal"
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-slate-800 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white"
        >
          Sign in
        </a>
      </header>
      <div className="space-y-10">
        {sections.map((section) => (
          <SectionView key={section.title} section={section} />
        ))}
      </div>
    </main>
  );
}

function SectionView({ section }: { section: Section }) {
  return (
    <section>
      <h2 className="mb-4 text-xl font-medium text-slate-700 dark:text-slate-300">
        {section.title}
      </h2>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {section.links.map((link) => (
          <LinkCard key={link.href} link={link} />
        ))}
      </div>
    </section>
  );
}

function LinkCard({ link }: { link: SectionLink }) {
  return (
    <a
      href={link.href}
      target="_blank"
      rel="noreferrer"
      className="block rounded-lg border border-slate-200 bg-white p-5 shadow-sm transition hover:border-slate-300 hover:shadow-md dark:border-slate-800 dark:bg-slate-900 dark:hover:border-slate-700"
    >
      <div className="text-base font-medium">{link.title}</div>
      <div className="mt-1 text-sm text-slate-600 dark:text-slate-400">
        {link.description}
      </div>
    </a>
  );
}
