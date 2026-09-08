"use server";

import { revalidatePath } from "next/cache";
import { auth } from "@/auth";
import { isPortalAuthorized } from "@/lib/authz";
import { clearCalibrationSource, setCalibrationSource } from "@/lib/dives";

export type ActionResult = { ok: true } | { ok: false; error: string };

/** Server actions are public endpoints — re-check on every call rather than
 * trusting that the client only renders them for permitted users.
 *
 * Checks authorization, not just authentication. The page-level guard is a
 * rendering decision; this is the one that actually protects the write, since
 * a server action can be invoked directly by anyone who can reach the app. */
async function requireAuthorized(): Promise<void> {
  const session = await auth();
  if (!session?.user) {
    throw new Error("Not authenticated");
  }
  if (!isPortalAuthorized(session)) {
    throw new Error("Not authorized");
  }
}

export async function setCalibrationSourceAction(
  diveId: number,
  sourceId: number,
): Promise<ActionResult> {
  try {
    await requireAuthorized();
    if (diveId === sourceId) {
      return { ok: false, error: "A dive cannot be its own calibration source" };
    }
    await setCalibrationSource(diveId, sourceId);
    revalidatePath("/portal");
    return { ok: true };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : "Failed" };
  }
}

export async function clearCalibrationSourceAction(
  diveId: number,
): Promise<ActionResult> {
  try {
    await requireAuthorized();
    await clearCalibrationSource(diveId);
    revalidatePath("/portal");
    return { ok: true };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : "Failed" };
  }
}
