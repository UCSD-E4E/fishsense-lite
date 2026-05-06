import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getIncompleteProjectIds } from "./fishsense-api";

beforeEach(() => {
  vi.stubEnv("FISHSENSE_API_URL", "http://api.test");
  vi.stubEnv("FISHSENSE_API_USERNAME", "alice");
  vi.stubEnv("FISHSENSE_API_PASSWORD", "secret");
  vi.stubEnv("LABEL_STUDIO_URL", "http://ls.test");
  vi.stubEnv("LABEL_STUDIO_API_KEY", "k");
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

describe("getIncompleteProjectIds", () => {
  it("calls all four label-kind endpoints with incomplete=true and Basic auth", async () => {
    const fetchMock = vi.fn<FetchSig>(async (url) => {
      if (url.includes("/labels/laser/")) return jsonResponse([42, 43]);
      if (url.includes("/labels/species/")) return jsonResponse([70]);
      if (url.includes("/labels/headtail/")) return jsonResponse([44, 45]);
      if (url.includes("/labels/dive-slate/")) return jsonResponse([66]);
      throw new Error(`unexpected url: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await getIncompleteProjectIds(60);

    expect(result).toEqual({
      laser: [42, 43],
      species: [70],
      headtail: [44, 45],
      "dive-slate": [66],
    });

    expect(fetchMock).toHaveBeenCalledTimes(4);
    const expectedAuth = `Basic ${Buffer.from("alice:secret").toString("base64")}`;
    for (const [url, init] of fetchMock.mock.calls) {
      expect(url).toMatch(/^http:\/\/api\.test\/api\/v1\/labels\//);
      expect(url).toContain("/label-studio-project-ids?incomplete=true");
      const headers = init?.headers as Record<string, string>;
      expect(headers.Authorization).toBe(expectedAuth);
    }
  });

  it("forwards revalidate to fetch's next option", async () => {
    const fetchMock = vi.fn<FetchSig>(async () => jsonResponse([]));
    vi.stubGlobal("fetch", fetchMock);

    await getIncompleteProjectIds(123);

    for (const [, init] of fetchMock.mock.calls) {
      expect(init?.next?.revalidate).toBe(123);
    }
  });

  it("throws when any endpoint returns non-OK", async () => {
    const fetchMock = vi.fn<FetchSig>(async (url) => {
      if (url.includes("/labels/laser/")) {
        return new Response("nope", { status: 500, statusText: "Server Error" });
      }
      return jsonResponse([]);
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(getIncompleteProjectIds(60)).rejects.toThrow(/laser.*500/);
  });

  it("hits all four endpoints in parallel (single Promise.all)", async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const fetchMock = vi.fn<FetchSig>(async () => {
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((r) => setTimeout(r, 5));
      inFlight -= 1;
      return jsonResponse([]);
    });
    vi.stubGlobal("fetch", fetchMock);

    await getIncompleteProjectIds(60);

    expect(maxInFlight).toBe(4);
  });
});
