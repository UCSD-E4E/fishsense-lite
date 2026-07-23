"use server";

import { revalidatePath } from "next/cache";
import { auth } from "@/auth";
import { clearCalibrationSource, setCalibrationSource } from "@/lib/dives";

export type ActionResult = { ok: true } | { ok: false; error: string };

/** Server actions are public endpoints — re-check the session on every call
 * rather than trusting that the client only renders for signed-in users. */
async function requireSession(): Promise<void> {
  const session = await auth();
  if (!session?.user) {
    throw new Error("Not authenticated");
  }
}

export async function setCalibrationSourceAction(
  diveId: number,
  sourceId: number,
): Promise<ActionResult> {
  try {
    await requireSession();
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
    await requireSession();
    await clearCalibrationSource(diveId);
    revalidatePath("/portal");
    return { ok: true };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : "Failed" };
  }
}
