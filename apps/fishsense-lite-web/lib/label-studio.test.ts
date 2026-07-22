import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { __resetTokenCache, getAccessToken, getProject, getProjects } from "./label-studio";

beforeEach(() => {
  vi.stubEnv("FISHSENSE_API_URL", "http://api.test");
  vi.stubEnv("FISHSENSE_API_USERNAME", "u");
  vi.stubEnv("FISHSENSE_API_PASSWORD", "p");
  vi.stubEnv("LABEL_STUDIO_URL", "http://ls.test");
  vi.stubEnv("LABEL_STUDIO_API_KEY", "ls-refresh-token");
  __resetTokenCache();
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  __resetTokenCache();
});

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    statusText: init.statusText ?? "OK",
    headers: { "content-type": "application/json", ...(init.headers ?? {}) },
  });
}

type NextFetchInit = RequestInit & { next?: { revalidate?: number } };
type FetchSig = (input: string, init?: NextFetchInit) => Promise<Response>;

const REFRESH_URL = "http://ls.test/api/token/refresh";

/** Routes token refreshes to a JWT and project GETs to `onProject`. */
function stubFetch(
  onProject: (url: string, init?: NextFetchInit) => Promise<Response>,
  accessToken = "jwt-abc",
) {
  const fetchMock = vi.fn<FetchSig>(async (url, init) => {
    if (url === REFRESH_URL) return jsonResponse({ access: accessToken });
    return onProject(url, init);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("getAccessToken", () => {
  it("exchanges the API key as a refresh token for an access JWT", async () => {
    const fetchMock = stubFetch(async () => jsonResponse({}));

    const token = await getAccessToken();

    expect(token).toBe("jwt-abc");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(REFRESH_URL);
    expect(init?.method).toBe("POST");
    // The configured key is the *refresh* token, not a bearer credential.
    expect(JSON.parse(String(init?.body))).toEqual({ refresh: "ls-refresh-token" });
  });

  it("caches the token instead of refreshing per request", async () => {
    const fetchMock = stubFetch(async (url) => {
      const id = Number(url.split("/").pop());
      return jsonResponse({ id, title: `p${id}` });
    });

    await getProjects([1, 2, 3], 60);

    const refreshCalls = fetchMock.mock.calls.filter(([url]) => url === REFRESH_URL);
    expect(refreshCalls).toHaveLength(1);
  });
});

describe("getProject", () => {
  it("calls /api/projects/{id} with Bearer JWT auth and returns id+title", async () => {
    const fetchMock = stubFetch(async () =>
      jsonResponse({ id: 42, title: "REEF Laser High Priority", extra: "ignored" }),
    );

    const project = await getProject(42, 60);

    expect(project).toEqual({ id: 42, title: "REEF Laser High Priority", isPublished: true });
    const projectCall = fetchMock.mock.calls.find(([url]) => url !== REFRESH_URL)!;
    expect(projectCall[0]).toBe("http://ls.test/api/projects/42");
    const headers = projectCall[1]?.headers as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer jwt-abc");
  });

  it("forwards revalidate to fetch's next option", async () => {
    const fetchMock = stubFetch(async () => jsonResponse({ id: 1, title: "x" }));

    await getProject(1, 999);

    const projectCall = fetchMock.mock.calls.find(([url]) => url !== REFRESH_URL)!;
    expect(projectCall[1]?.next?.revalidate).toBe(999);
  });

  it("refreshes once and retries when a cached JWT has gone stale", async () => {
    let projectCalls = 0;
    const fetchMock = stubFetch(async () => {
      projectCalls += 1;
      if (projectCalls === 1) {
        return new Response("stale", { status: 401, statusText: "Unauthorized" });
      }
      return jsonResponse({ id: 7, title: "recovered" });
    });

    const project = await getProject(7, 60);

    expect(project).toEqual({ id: 7, title: "recovered", isPublished: true });
    const refreshCalls = fetchMock.mock.calls.filter(([url]) => url === REFRESH_URL);
    expect(refreshCalls).toHaveLength(2); // initial + forced retry
  });

  it("throws on non-OK response with status in the message", async () => {
    stubFetch(async () => new Response("nope", { status: 404, statusText: "Not Found" }));

    await expect(getProject(99, 60)).rejects.toThrow(/Label Studio project 99.*404/);
  });
});

describe("getProjects", () => {
  it("returns an empty list when given no IDs without calling fetch", async () => {
    const fetchMock = vi.fn<FetchSig>();
    vi.stubGlobal("fetch", fetchMock);

    const projects = await getProjects([], 60);

    expect(projects).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("fetches each ID in parallel and preserves order", async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    stubFetch(async (url) => {
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((r) => setTimeout(r, 5));
      inFlight -= 1;
      const id = Number(url.split("/").pop());
      return jsonResponse({ id, title: `project-${id}` });
    });

    const projects = await getProjects([1, 2, 3], 60);

    expect(projects).toEqual([
      { id: 1, title: "project-1", isPublished: true },
      { id: 2, title: "project-2", isPublished: true },
      { id: 3, title: "project-3", isPublished: true },
    ]);
    expect(maxInFlight).toBe(3);
  });

  it("drops unresolvable IDs instead of failing the whole page", async () => {
    // The prod shape: fishsense-api still hands out legacy ids (57-117) from
    // the retired self-hosted instance and every one 404s. Under Promise.all
    // one of these 500'd the entire landing page.
    stubFetch(async (url) => {
      const id = Number(url.split("/").pop());
      if (id === 73) return new Response("gone", { status: 404, statusText: "Not Found" });
      return jsonResponse({ id, title: `project-${id}` });
    });

    const projects = await getProjects([73, 274558], 60);

    expect(projects).toEqual([{ id: 274558, title: "project-274558", isPublished: true }]);
  });

  it("returns an empty list rather than throwing when every ID is dead", async () => {
    stubFetch(async () => new Response("gone", { status: 404, statusText: "Not Found" }));

    await expect(getProjects([57, 73, 117], 60)).resolves.toEqual([]);
  });
});
