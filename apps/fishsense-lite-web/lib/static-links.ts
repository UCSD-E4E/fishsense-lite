export type StaticLink = {
  title: string;
  description: string;
  href: string;
};

// Overridable via env so a host change doesn't need a code deploy.
const ANALYTICS_BASE =
  process.env.ANALYTICS_BASE ??
  "https://analytics.fishsense.e4e.ucsd.edu/superset/dashboard";

// Temporal moved to the shared krg-prod cluster at the Incus migration; the
// old in-orchestrator `workflows.fishsense.e4e.ucsd.edu` UI is gone. Set
// WORKFLOWS_URL to the current UI (if any is exposed to tenants) — the
// default below is the retired host and only a placeholder.
const WORKFLOWS_URL =
  process.env.WORKFLOWS_URL ?? "https://workflows.fishsense.e4e.ucsd.edu/";

export const RESULTS_LINKS: StaticLink[] = [
  {
    title: "Lengths",
    description: "Lengths dashboard from Apache Superset",
    href: `${ANALYTICS_BASE}/reef-smile-lengths`,
  },
  {
    title: "Metrics",
    description: "Metrics dashboard from Apache Superset",
    href: `${ANALYTICS_BASE}/reef-smile-metrics`,
  },
];

export const ADMIN_LINKS: StaticLink[] = [
  {
    title: "Workflows",
    description: "Temporal Workflows",
    href: WORKFLOWS_URL,
  },
];
