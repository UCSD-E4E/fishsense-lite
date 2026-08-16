import { fishsenseApiAuthHeader } from "./api-auth";
import { env } from "./env";

type LabelKind = "laser" | "species" | "headtail" | "dive-slate";

async function getProjectIds(kind: LabelKind, revalidate: number): Promise<number[]> {
  // `LabelKind` values are the URL segments verbatim — there was a
  // `KIND_TO_PATH` map here that mapped each key to itself.
  const url = `${env.fishsenseApiUrl}/api/v1/labels/${kind}/label-studio-project-ids?incomplete=true`;

  const response = await fetch(url, {
    headers: { Authorization: fishsenseApiAuthHeader() },
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
