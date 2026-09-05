import { env } from "./env";
import { getAccessToken } from "./label-studio";
import type { LsRegion, LsTask } from "./triage";

export type TaskPage = {
  tasks: LsTask[];
  total: number;
  /** True when the page is past the end of the result set. */
  drained: boolean;
};

/**
 * Authenticated Label Studio request, refreshing the access token once on 401.
 *
 * The configured key is a *personal access token* — itself a refresh token,
 * good for a JWT that lives about five minutes. That is shorter than a
 * labeling session, so every call has to be able to re-mint mid-flight. Doing
 * it here rather than per-caller is why nothing above this layer knows the
 * token exists.
 */
async function authed(path: string, init: RequestInit = {}): Promise<Response> {
  const url = `${env.labelStudioUrl}${path}`;
  const send = async (token: string) =>
    fetch(url, {
      ...init,
      headers: {
        ...(init.headers ?? {}),
        Authorization: `Bearer ${token}`,
      },
      cache: "no-store",
    });

  const response = await send(await getAccessToken());
  if (response.status === 401 || response.status === 403) {
    return send(await getAccessToken(true));
  }
  return response;
}

const DEFAULT_PAGE_SIZE = 50;

/**
 * One page of a project's tasks, with predictions and annotations inlined.
 *
 * A 404 here is not a failure. DRF answers a page past the end of a result set
 * with `{"detail": "Invalid page."}` and status 404 rather than an empty list,
 * so a paging loop that treats it as an error dies mid-scan instead of moving
 * on to the next project.
 */
export async function listTasks(
  projectId: number,
  page: number,
  pageSize: number = DEFAULT_PAGE_SIZE,
): Promise<TaskPage> {
  const response = await authed(
    `/api/tasks/?project=${projectId}&page=${page}&page_size=${pageSize}`,
  );

  if (response.status === 404) {
    return { tasks: [], total: 0, drained: true };
  }
  if (!response.ok) {
    throw new Error(
      `Label Studio tasks fetch failed for project ${projectId} page ${page}: ` +
        `${response.status} ${response.statusText}`,
    );
  }

  const body = (await response.json()) as
    | LsTask[]
    | { tasks?: LsTask[]; total?: number };

  // Older instances return a bare array; newer ones wrap it.
  const tasks = Array.isArray(body) ? body : (body.tasks ?? []);
  const total = Array.isArray(body) ? body.length : (body.total ?? tasks.length);
  return { tasks, total, drained: tasks.length === 0 };
}

/**
 * Accept: store the prediction's regions verbatim as a human annotation.
 *
 * `result` is passed through untouched and MUST be the array read straight off
 * the prediction. The sync activity reads `original_width`, `original_height`,
 * `value.x`, `value.y` and `value.keypointlabels[0]` back out of what is
 * stored here, so copying is correct by construction where rebuilding is a new
 * chance to be wrong.
 *
 * `lead_time` is Label Studio's own "seconds spent on this task" field. It is
 * the cheapest signal available for rubber-stamping — a labeler averaging a
 * few hundred milliseconds is not inspecting anything — so it is always sent.
 *
 * There is deliberately no skip counterpart. See the note in `triage.ts`.
 */
export async function acceptPrediction(
  taskId: number,
  result: LsRegion[],
  leadTimeMs: number,
): Promise<number> {
  const response = await authed(`/api/tasks/${taskId}/annotations/`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      result,
      lead_time: leadTimeMs,
      was_cancelled: false,
      // Label Studio stamps `origin: "prediction"` on the stored regions
      // itself, which is what the auto-accept gate also produces. Nothing
      // downstream distinguishes the two, by design.
    }),
  });

  if (!response.ok) {
    throw new Error(
      `Label Studio annotation POST failed for task ${taskId}: ` +
        `${response.status} ${response.statusText}`,
    );
  }

  const data = (await response.json()) as { id?: number };
  return data.id ?? 0;
}

/** Delete an annotation — used to undo an accept made moments ago. */
export async function deleteAnnotation(annotationId: number): Promise<void> {
  const response = await authed(`/api/annotations/${annotationId}/`, { method: "DELETE" });
  if (!response.ok && response.status !== 404) {
    throw new Error(
      `Label Studio annotation DELETE failed for ${annotationId}: ` +
        `${response.status} ${response.statusText}`,
    );
  }
}

/**
 * Stream a task's image through this server.
 *
 * Tasks hold `s3://` URIs. With `resolve_uri=true` Label Studio does not hand
 * back a presigned S3 URL — it returns a path on its own API server, which is
 * **relative** and **authenticated**. Requiring `https://` rejected every task
 * in the Android app and reported "queue empty" against a project holding 283.
 *
 * Proxying is what keeps that off the client: the browser asks this server for
 * the bytes and never needs a Label Studio credential, which also sidesteps
 * the five-minute bearer expiring mid-session.
 */
export async function fetchTaskImage(taskId: number, imageUri: string): Promise<Response> {
  return authed(`/tasks/${taskId}/resolve/?fileuri=${encodeURIComponent(base64(imageUri))}`);
}

/** `fileuri` is the base64 of the stored `s3://` URI, URL-encoded. */
export function base64(value: string): string {
  return Buffer.from(value, "utf8").toString("base64");
}

/** One task, with its predictions and annotations. */
export async function getTask(taskId: number): Promise<LsTask | null> {
  const response = await authed(`/api/tasks/${taskId}/`);
  if (response.status === 404) return null;
  if (!response.ok) {
    throw new Error(
      `Label Studio task ${taskId} fetch failed: ${response.status} ${response.statusText}`,
    );
  }
  return (await response.json()) as LsTask;
}
