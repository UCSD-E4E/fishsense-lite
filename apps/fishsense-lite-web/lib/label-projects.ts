import { labelStudioEnabled } from "./env";
import { getProjectIds, type LabelKind } from "./fishsense-api";
import type { LabelStudioProject } from "./label-studio";

/**
 * Which Label Studio projects a human should be sent to, for one label kind.
 *
 * **One definition, two consumers.** The landing page renders these as cards
 * and triage walks their tasks; before this existed each answered the question
 * for itself and they drifted. The gate filter reached triage by inheritance
 * rather than by anyone deciding it should, and the Label Studio kill switch
 * reached it not at all — turning `LABEL_STUDIO_ENABLED` off blanked the
 * landing page while triage carried on calling Label Studio.
 *
 * Returns ids rather than resolved projects on purpose: the two consumers need
 * genuinely different fetch strategies — the landing page resolves every
 * project to title them, triage resolves them lazily and stops as soon as it
 * has a batch — and that difference is legitimate. What is not legitimate is
 * each one re-deciding *which* projects count.
 *
 * Order is NOT part of the policy: the landing page shows cards in the order
 * the API returns, and reordering them here would change that surface for a
 * reason belonging to the other one. Triage sorts for itself.
 */
export async function liveProjectIds(
  kind: LabelKind,
  revalidate: number,
): Promise<number[]> {
  // Nothing to send anyone to when the integration is switched off. This is
  // the check triage never had.
  if (!labelStudioEnabled()) return [];
  return getProjectIds(kind, revalidate);
}

/**
 * Does this project still hold labeling a human should do?
 *
 * **One question, one answer, one source.** Both surfaces ask it: the landing
 * page to decide whether to link a project, triage to decide whether to walk
 * it. They used to answer it separately and from different systems — the
 * landing page from `LaserLabel.completed` in our own database, triage from
 * Label Studio task state — and so they disagreed, listing nine projects with
 * "outstanding work" that held nothing anyone could act on.
 *
 * Label Studio is the source. Our `completed` column is a mirror of it,
 * written by the hourly sync, so reading the mirror shows a stale answer
 * whenever the sync is behind — and a permanently wrong one for any row the
 * sync never revisits.
 *
 * Fails OPEN. When Label Studio does not report the counts we keep the
 * project, because hiding real labeling work is the worse error: a labeler
 * cannot act on a queue that does not mention it, and nothing else would
 * report the omission.
 */
export function hasPendingWork(project: LabelStudioProject): boolean {
  const { taskCount, annotatedCount } = project;
  if (taskCount === undefined || annotatedCount === undefined) return true;
  return taskCount - annotatedCount > 0;
}

/**
 * Drop unpublished projects.
 *
 * The ids come from fishsense-api, which is derived from label rows and knows
 * nothing about Label Studio's publish state — so a draft still being
 * populated, or one deliberately held back, would otherwise be offered.
 *
 * `!== false` rather than truthiness: only an explicit unpublished flag hides
 * a project, so a Label Studio response change cannot silently blank
 * everything.
 */
export function isPublished(project: LabelStudioProject): boolean {
  return project.isPublished !== false;
}
