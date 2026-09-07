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
const RATE_LIMIT_RETRIES = 3;
const RATE_LIMIT_BASE_MS = 750;

/** Hosted Label Studio rate-limits, and says how long to wait when it does. */
function retryAfterMs(response: Response, attempt: number): number {
  const header = response.headers.get("retry-after");
  if (header) {
    const seconds = Number(header);
    if (Number.isFinite(seconds) && seconds >= 0) return seconds * 1000;
    const at = Date.parse(header);
    if (Number.isFinite(at)) return Math.max(0, at - Date.now());
  }
  // Exponential, so a burst backs off rather than re-forming.
  return RATE_LIMIT_BASE_MS * 2 ** attempt;
}

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

  let response = await send(await getAccessToken());
  if (response.status === 401 || response.status === 403) {
    response = await send(await getAccessToken(true));
  }

  // 429 is a "come back shortly", not a failure — surfacing it aborts the
  // whole queue load over a condition that clears on its own. Honour
  // `Retry-After` when the server sends one; it knows the window and we do
  // not.
  for (let attempt = 0; response.status === 429 && attempt < RATE_LIMIT_RETRIES; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, retryAfterMs(response, attempt)));
    response = await send(await getAccessToken());
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
 * Fetch a task's frame, letting Label Studio say where it lives.
 *
 * The URL is NOT constructed here any more. It used to be built by hand as
 * `/tasks/{id}/resolve/?fileuri={base64}`, copied from a note about hosted
 * Label Studio — and every fetch came back non-OK, which the route turned into
 * a bare 502 with nothing to diagnose from.
 *
 * The supported route is to ask: request the task with `resolve_uri=true` and
 * read whatever `data.image` becomes. What comes back varies by deployment and
 * by how the project's storage is configured, so all three shapes are handled:
 *
 *   * an absolute `http(s)` URL — a presigned link, fetched WITHOUT our
 *     Authorization header, because sending a bearer to S3 can itself be
 *     rejected;
 *   * a root-relative path on Label Studio's own server — fetched WITH auth,
 *     since that endpoint is authenticated;
 *   * still `s3://` — Label Studio could not resolve it, which means the
 *     project has no storage connected, and no amount of fetching will help.
 *     Reported as such rather than retried.
 */
export type ResolvedImage =
  | { kind: "response"; response: Response; url: string }
  | { kind: "unresolved"; uri: string };

export async function fetchTaskImage(taskId: number): Promise<ResolvedImage> {
  const task = await getTask(taskId, { resolveUri: true });
  const uri = typeof task?.data?.image === "string" ? task.data.image : "";

  if (!uri || uri.startsWith("s3://") || uri.startsWith("gs://")) {
    return { kind: "unresolved", uri };
  }

  if (/^https?:\/\//i.test(uri)) {
    // Presigned: the signature IS the credential, and adding ours can trip
    // S3's "only one auth mechanism" rule.
    return { kind: "response", response: await fetch(uri, { cache: "no-store" }), url: uri };
  }

  const path = uri.startsWith("/") ? uri : `/${uri}`;
  return { kind: "response", response: await authed(path), url: path };
}

/** One task, with its predictions and annotations. */
export async function getTask(
  taskId: number,
  { resolveUri = false }: { resolveUri?: boolean } = {},
): Promise<LsTask | null> {
  // `resolve_uri=true` asks Label Studio to rewrite storage URIs in `data`
  // into something fetchable. It is off by default because the queue only
  // needs predictions, and resolving costs the server work per task.
  const query = resolveUri ? "?resolve_uri=true" : "";
  const response = await authed(`/api/tasks/${taskId}/${query}`);
  if (response.status === 404) return null;
  if (!response.ok) {
    throw new Error(
      `Label Studio task ${taskId} fetch failed: ${response.status} ${response.statusText}`,
    );
  }
  return (await response.json()) as LsTask;
}
