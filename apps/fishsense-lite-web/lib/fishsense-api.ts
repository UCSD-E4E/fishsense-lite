import { env } from "./env";

type LabelKind = "laser" | "species" | "headtail" | "dive-slate";

const KIND_TO_PATH: Record<LabelKind, string> = {
  laser: "laser",
  species: "species",
  headtail: "headtail",
  "dive-slate": "dive-slate",
};

async function getProjectIds(kind: LabelKind, revalidate: number): Promise<number[]> {
  const path = KIND_TO_PATH[kind];
  const url = `${env.fishsenseApiUrl}/api/v1/labels/${path}/label-studio-project-ids?incomplete=true`;
  const auth = Buffer.from(
    `${env.fishsenseApiUsername}:${env.fishsenseApiPassword}`,
  ).toString("base64");

  const response = await fetch(url, {
    headers: { Authorization: `Basic ${auth}` },
    next: { revalidate },
  });

  if (!response.ok) {
    console.error(
      `[fishsense-api] ${kind} project IDs fetch failed`,
      { url, status: response.status, statusText: response.statusText },
    );
    throw new Error(
      `fishsense-api ${kind} project IDs failed: ${response.status} ${response.statusText}`,
    );
  }

  return (await response.json()) as number[];
}

export async function getIncompleteProjectIds(
  revalidate: number,
): Promise<Record<LabelKind, number[]>> {
  const [laser, species, headtail, slate] = await Promise.all([
    getProjectIds("laser", revalidate),
    getProjectIds("species", revalidate),
    getProjectIds("headtail", revalidate),
    getProjectIds("dive-slate", revalidate),
  ]);
  return { laser, species, headtail, "dive-slate": slate };
}

export type { LabelKind };
