import { getIncompleteProjectIds } from "./fishsense-api";
import { getProjects, type LabelStudioProject } from "./label-studio";
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

  let outstanding: number[] | null = null;
  try {
    const ids = await getIncompleteProjectIds(revalidate);
    outstanding = kindKey === "laser" ? ids.laser : ids.headtail;
  } catch {
    outstanding = null;
  }
  if (!outstanding || outstanding.length === 0) {
    return { items: [], scanned: 0, rejected: [] };
  }

  const projects: LabelStudioProject[] = (await getProjects(outstanding, revalidate))
    .filter((p) => p.isPublished !== false)
    .sort((a, b) => b.id - a.id);

  const items: TriageItem[] = [];
  const rejected: string[] = [];
  let scanned = 0;

  for (const project of projects.slice(0, MAX_PROJECTS_PER_LOAD)) {
    if (items.length >= want) break;
    scanned += 1;

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
