import { fishsenseApiAuthHeader } from "./api-auth";
import { env } from "./env";

type LabelKind = "laser" | "species" | "headtail" | "dive-slate";

// Kinds whose predictions carry an auto-accept gate verdict.
//
// Only laser has one: `LaserPrediction.gate_verdict` is the sole gate field
// in the API's models. Add a kind here when its prediction model grows one —
// head/tail is the next candidate, since it is getting predictions now.
//
// Sending `gated=true` to a kind with no gate would ask for a condition
// nothing can satisfy and silently blank that section of the landing page,
// which is why this is a per-kind opt-in rather than a blanket parameter.
const GATED_KINDS: ReadonlySet<LabelKind> = new Set<LabelKind>(["laser"]);

export async function getProjectIds(kind: LabelKind, revalidate: number): Promise<number[]> {
  // `LabelKind` values are the URL segments verbatim — there was a
  // `KIND_TO_PATH` map here that mapped each key to itself.
  //
  // `gated=true` hides projects the auto-accept gate has not finished with.
  // Those projects' pending frames are the machine's work, not a labeler's:
  // the gate is about to accept them, so a human who judges them first has
  // done the work twice. The API's predicate is "the gate is done here", not
  // "the gate has run here" — a half-swept dive still holds frames it is
  // about to take.
  const params = new URLSearchParams({ incomplete: "true" });
  if (GATED_KINDS.has(kind)) {
    params.set("gated", "true");
  }
  const url = `${env.fishsenseApiUrl}/api/v1/labels/${kind}/label-studio-project-ids?${params}`;

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
