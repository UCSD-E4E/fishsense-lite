import { isPublished, liveProjectIds } from "./label-projects";
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

/**
 * What one project contributed, and why.
 *
 * Per project rather than a flat sample. The first version capped the whole
 * scan at five reasons, so the first project consumed every slot and every
 * later project's outcome was invisible — which made a project that WAS walked
 * look like one that was never offered.
 */
export type ProjectOutcome = {
  projectId: number;
  title?: string;
  /** Tasks the project returned on the first page. */
  tasks: number;
  /** Triageable items taken from it. */
  taken: number;
  /** Refusal reason -> how many tasks it applied to. */
  reasons: Record<string, number>;
  /** Set when the project itself could not be read. */
  error?: string;
};

export type QueueReport = {
  items: TriageItem[];
  scanned: number;
  projects: ProjectOutcome[];
  /** Projects discovery offered but the walk did not reach. */
  notWalked: number;
};

/** Strips task ids so reasons group: "task 41: is_labeled" -> "is_labeled". */
export function reasonKey(reason: string): string {
  return reason.replace(/^task \d+: /, "");
}

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
): Promise<QueueReport> {
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
    return { items: [], scanned: 0, projects: [], notWalked: 0 };
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
  const projects: ProjectOutcome[] = [];
  let scanned = 0;

  for (const projectId of candidates.slice(0, MAX_PROJECTS_PER_LOAD)) {
    if (items.length >= want) break;
    scanned += 1;

    const outcome: ProjectOutcome = { projectId, tasks: 0, taken: 0, reasons: {} };
    projects.push(outcome);

    let project;
    try {
      project = await getProject(projectId, revalidate);
    } catch (error) {
      // A legacy id that no longer resolves must not take down the page.
      outcome.error = error instanceof Error ? error.message : "unresolvable";
      continue;
    }
    outcome.title = project.title;
    if (!isPublished(project)) {
      outcome.error = "unpublished in Label Studio";
      continue;
    }

    const page = await listTasks(project.id, 1);
    outcome.tasks = page.tasks.length;
    for (const task of page.tasks) {
      if (items.length >= want) break;
      const reason = rejectionReason(task, kind, EMPTY);
      if (reason) {
        const key = reasonKey(reason);
        outcome.reasons[key] = (outcome.reasons[key] ?? 0) + 1;
        continue;
      }
      outcome.taken += 1;
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

  return {
    items,
    scanned,
    projects,
    notWalked: Math.max(0, candidates.length - scanned),
  };
}

/** Server-side load knows nothing about this session's skips — the client
 *  filters those out itself, so the same task is never re-shown. */
const EMPTY: ReadonlySet<number> = new Set<number>();

export { isTriageable };
