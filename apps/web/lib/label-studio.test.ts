import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getProject, getProjects } from "./label-studio";

beforeEach(() => {
  vi.stubEnv("FISHSENSE_API_URL", "http://api.test");
  vi.stubEnv("FISHSENSE_API_USERNAME", "u");
  vi.stubEnv("FISHSENSE_API_PASSWORD", "p");
  vi.stubEnv("LABEL_STUDIO_URL", "http://ls.test");
  vi.stubEnv("LABEL_STUDIO_API_KEY", "ls-key-123");
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
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

describe("getProject", () => {
  it("calls /api/projects/{id} with Token auth and returns id+title", async () => {
    const fetchMock = vi.fn<FetchSig>(async () =>
      jsonResponse({ id: 42, title: "REEF Laser High Priority", extra: "ignored" }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const project = await getProject(42, 60);

    expect(project).toEqual({ id: 42, title: "REEF Laser High Priority" });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://ls.test/api/projects/42");
    const headers = init?.headers as Record<string, string>;
    expect(headers.Authorization).toBe("Token ls-key-123");
  });

  it("forwards revalidate to fetch's next option", async () => {
    const fetchMock = vi.fn<FetchSig>(async () => jsonResponse({ id: 1, title: "x" }));
    vi.stubGlobal("fetch", fetchMock);

    await getProject(1, 999);

    const [, init] = fetchMock.mock.calls[0];
    expect(init?.next?.revalidate).toBe(999);
  });

  it("throws on non-OK response with status in the message", async () => {
    const fetchMock = vi.fn<FetchSig>(
      async () => new Response("nope", { status: 404, statusText: "Not Found" }),
    );
    vi.stubGlobal("fetch", fetchMock);

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
    const fetchMock = vi.fn<FetchSig>(async (url) => {
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((r) => setTimeout(r, 5));
      inFlight -= 1;
      const id = Number(url.split("/").pop());
      return jsonResponse({ id, title: `project-${id}` });
    });
    vi.stubGlobal("fetch", fetchMock);

    const projects = await getProjects([1, 2, 3], 60);

    expect(projects).toEqual([
      { id: 1, title: "project-1" },
      { id: 2, title: "project-2" },
      { id: 3, title: "project-3" },
    ]);
    expect(maxInFlight).toBe(3);
  });
});
