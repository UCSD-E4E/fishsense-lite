import type { ActiveProjects } from "./active-projects";
import { LABELER_BASE, type StaticLink } from "./static-links";

export type SectionLink = {
  title: string;
  description: string;
  href: string;
};

export type Section = {
  title: string;
  links: SectionLink[];
};

const LABELING_KINDS: { key: keyof ActiveProjects; title: string }[] = [
  { key: "laser", title: "Laser Labeling" },
  { key: "headtail", title: "Head/Tail Labeling" },
  { key: "species", title: "Species Labeling" },
  { key: "slate", title: "Slate Labeling" },
];

export function buildSections(
  active: ActiveProjects,
  staticLinks: { results: StaticLink[]; admin: StaticLink[] },
): Section[] {
  const sections: Section[] = [];

  for (const { key, title } of LABELING_KINDS) {
    const projects = active[key];
    if (projects.length === 0) continue;
    sections.push({
      title,
      links: projects.map((p) => ({
        title: p.title,
        description: `${p.title} labeling project`,
        href: `${LABELER_BASE}/projects/${p.id}`,
      })),
    });
  }

  if (staticLinks.results.length > 0) {
    sections.push({ title: "Results", links: staticLinks.results });
  }
  if (staticLinks.admin.length > 0) {
    sections.push({ title: "Administration", links: staticLinks.admin });
  }

  return sections;
}
