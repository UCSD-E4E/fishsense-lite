import { hasPendingWork, isPublished, liveProjectIds } from "./label-projects";
import { getProject } from "./label-studio";
import { listTasks } from "./label-studio-tasks";
import {
  QUEUE_KINDS,
  diveNameFromTitle,
  isTriageable,
  keypointsOf,
  pickPrediction,
  rejectionReason,
  type Keypoint,
  type LsRegion,
  type QueueKind,
} from "./triage";

export type TriageItem = {
  taskId: number;
  projectId: number;
  diveName: string;
  keypoints: Keypoint[];
  /** Passed back verbatim on accept — never rebuilt. */
  result: LsRegion[];
  partial: boolean;
};

/** How many projects to walk before giving up on filling a batch. */
const MAX_PROJECTS_PER_LOAD = 12;

/**
 * The next batch of triageable tasks for a kind, newest dive first.
 *
 * Discovery goes through fishsense-api rather than Label Studio: one request
 * names the projects that still hold outstanding label rows, which is
 * authoritative where anything derived from Label Studio is inferred. When the
 * API cannot answer we keep every project — treating "cannot ask" as "nothing
 * to do" would report an empty queue while Label Studio is full of work, which
 * is the same silent-loss shape as writing on skip.
 */
export async function loadQueue(
  kindKey: QueueKind["key"],
  want = 12,
  revalidate = 0,
): Promise<{ items: TriageItem[]; scanned: number; rejected: string[] }> {
  const kind = QUEUE_KINDS[kindKey];

  // Deliberately NOT wrapped in a try/catch.
  //
  // It used to swallow any discovery failure into an empty queue, which is the
  // exact silent-loss shape this feature is otherwise careful to avoid:
  // treating "cannot ask" as "nothing to do" reports a drained queue while
  // Label Studio is full of work, and looks identical to the legitimate empty
  // state. A failure here should reach the page and be read.
  const outstanding = await liveProjectIds("laser", revalidate);
  if (outstanding.length === 0) {
    return { items: [], scanned: 0, rejected: [] };
  }

  // Resolved ONE AT A TIME, in the order the policy returned.
  //
  // This used to resolve every outstanding project up front — dozens of
  // parallel calls to hosted Label Studio on page load, which earned a 429 —
  // and then walk at most a handful. Sequential and lazy: two requests per
  // project actually walked, stopping as soon as there is a batch.
  // Newest dive first: project ids ascend with dives, and the recent ones are
  // where labeling is still happening. Sorted here rather than in the shared
  // policy, because the landing page has its own order.
  const candidates = [...outstanding].sort((a, b) => b - a);

  const items: TriageItem[] = [];
  const rejected: string[] = [];
  let scanned = 0;

  for (const projectId of candidates.slice(0, MAX_PROJECTS_PER_LOAD)) {
    if (items.length >= want) break;
    scanned += 1;

    let project;
    try {
      project = await getProject(projectId, revalidate);
    } catch (error) {
      // A legacy id that no longer resolves must not take down the page.
      if (rejected.length < 5) {
        rejected.push(`project ${projectId}: ${error instanceof Error ? error.message : "unresolvable"}`);
      }
      continue;
    }
    if (!isPublished(project)) continue;

    const page = await listTasks(project.id, 1);
    for (const task of page.tasks) {
      if (items.length >= want) break;
      const reason = rejectionReason(task, kind, EMPTY);
      if (reason) {
        if (rejected.length < 5) rejected.push(reason);
        continue;
      }
      const prediction = pickPrediction(task)!;
      const keypoints = keypointsOf(prediction);
      items.push({
        taskId: task.id,
        projectId: project.id,
        diveName: diveNameFromTitle(project.title, kind),
        keypoints,
        result: prediction.result,
        partial: keypoints.length < kind.expectedKeypoints,
      });
    }
  }

  return { items, scanned, rejected };
}

/** Server-side load knows nothing about this session's skips — the client
 *  filters those out itself, so the same task is never re-shown. */
const EMPTY: ReadonlySet<number> = new Set<number>();

export { isTriageable };
