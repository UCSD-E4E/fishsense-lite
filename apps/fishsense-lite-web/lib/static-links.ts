export type StaticLink = {
  title: string;
  description: string;
  href: string;
};

const ANALYTICS_BASE = "https://analytics.fishsense.e4e.ucsd.edu/superset/dashboard";

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
    href: "https://workflows.fishsense.e4e.ucsd.edu/",
  },
];

export const LABELER_BASE = "https://labeler.e4e.ucsd.edu";
