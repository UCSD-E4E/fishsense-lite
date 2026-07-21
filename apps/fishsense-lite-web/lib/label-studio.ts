import { env } from "./env";

export type LabelStudioProject = {
  id: number;
  title: string;
};

// Hosted Label Studio (app.heartex.com) does NOT accept the configured key
// as a bearer credential. `LABEL_STUDIO_API_KEY` is a *personal access
// token* — a refresh token — which must be exchanged at
// `/api/token/refresh` for a short-lived access JWT that is then sent as
// `Authorization: Bearer <jwt>`.
//
// This is why the integration was originally kill-switched off: the app
// sent `Authorization: Token <key>`, which 401s on every request. Verified
// against prod 2026-07-21 — `Token <key>` and `Bearer <key>` both 401,
// while refresh -> `Bearer <jwt>` returns 200.
const DEFAULT_TOKEN_TTL_SECONDS = 240;
const EXPIRY_SKEW_SECONDS = 30;

type CachedToken = { token: string; expiresAtMs: number };
let cachedToken: CachedToken | null = null;
// Deduplicates concurrent refreshes. `getProjects` fans out over every id at
// once, so without this each one races on an empty cache and fires its own
// refresh POST — a dozen redundant round trips per page render.
let inFlightRefresh: Promise<string> | null = null;

/** Seconds until a JWT expires, from its `exp` claim; null if unreadable. */
function jwtLifetimeSeconds(token: string): number | null {
  const payload = token.split(".")[1];
  if (!payload) return null;
  try {
    const json = JSON.parse(
      Buffer.from(payload.replace(/-/g, "+").replace(/_/g, "/"), "base64").toString(
        "utf8",
      ),
    ) as { exp?: number };
    if (typeof json.exp !== "number") return null;
    return json.exp - Math.floor(Date.now() / 1000);
  } catch {
    return null;
  }
}

async function refreshAccessToken(): Promise<string> {
  const url = `${env.labelStudioUrl}/api/token/refresh`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ refresh: env.labelStudioApiKey }),
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `Label Studio token refresh failed: ${response.status} ${response.statusText}`,
    );
  }

  const data = (await response.json()) as { access?: string };
  if (!data.access) {
    throw new Error("Label Studio token refresh returned no access token");
  }

  const lifetime = jwtLifetimeSeconds(data.access) ?? DEFAULT_TOKEN_TTL_SECONDS;
  const ttl = Math.max(lifetime - EXPIRY_SKEW_SECONDS, 1);
  cachedToken = { token: data.access, expiresAtMs: Date.now() + ttl * 1000 };
  return data.access;
}

export async function getAccessToken(forceRefresh = false): Promise<string> {
  if (!forceRefresh && cachedToken && cachedToken.expiresAtMs > Date.now()) {
    return cachedToken.token;
  }
  if (!forceRefresh && inFlightRefresh) {
    return inFlightRefresh;
  }

  const pending = refreshAccessToken();
  inFlightRefresh = pending;
  try {
    return await pending;
  } finally {
    if (inFlightRefresh === pending) {
      inFlightRefresh = null;
    }
  }
}

/** Test seam — drops the cached access token and any in-flight refresh. */
export function __resetTokenCache(): void {
  cachedToken = null;
  inFlightRefresh = null;
}

export async function getProject(
  id: number,
  revalidate: number,
): Promise<LabelStudioProject> {
  const url = `${env.labelStudioUrl}/api/projects/${id}`;

  const attempt = async (token: string) =>
    fetch(url, {
      headers: { Authorization: `Bearer ${token}` },
      next: { revalidate },
    });

  let response = await attempt(await getAccessToken());
  if (response.status === 401 || response.status === 403) {
    // Cached JWT went stale early (or was revoked) — refresh once.
    response = await attempt(await getAccessToken(true));
  }

  if (!response.ok) {
    console.error(`[label-studio] project ${id} fetch failed`, {
      url,
      status: response.status,
      statusText: response.statusText,
    });
    throw new Error(
      `Label Studio project ${id} fetch failed: ${response.status} ${response.statusText}`,
    );
  }

  const data = (await response.json()) as { id: number; title: string };
  return { id: data.id, title: data.title };
}

export async function getProjects(
  ids: number[],
  revalidate: number,
): Promise<LabelStudioProject[]> {
  // Tolerate individual failures. fishsense-api still stores legacy project
  // ids (57-117) from the retired self-hosted instance, and every one of
  // them 404s on the hosted one. Under `Promise.all` a single dead id
  // rejected out of the server component and 500'd the entire landing page,
  // which is the other half of why this integration got kill-switched off.
  // Drop what we can't resolve and render the rest.
  const settled = await Promise.allSettled(ids.map((id) => getProject(id, revalidate)));

  const projects: LabelStudioProject[] = [];
  const failedIds: number[] = [];
  settled.forEach((result, index) => {
    if (result.status === "fulfilled") {
      projects.push(result.value);
    } else {
      failedIds.push(ids[index]);
    }
  });

  if (failedIds.length > 0) {
    console.warn(
      `[label-studio] skipped ${failedIds.length} unresolvable project id(s): ${failedIds.join(", ")}`,
    );
  }

  return projects;
}
