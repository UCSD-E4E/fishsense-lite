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

describe("the auto-accept gate filter", () => {
  // The gate decides which laser predictions a human never needs to see.
  // Surfacing a project it has not finished with sends a labeler at frames
  // the machine is about to accept for them — duplicated work, and worse,
  // work that looks voluntary. `gated=true` means "the gate is done here",
  // not merely "the gate has run here"; the API owns that distinction.
  it("asks only for gated laser projects", async () => {
    const fetchMock = vi.fn<FetchSig>(async () => jsonResponse([]));
    vi.stubGlobal("fetch", fetchMock);

    await getIncompleteProjectIds(60);

    const laserCall = fetchMock.mock.calls.find(([url]) =>
      url.includes("/labels/laser/"),
    );
    expect(laserCall?.[0]).toContain("gated=true");
  });

  // Only `LaserPrediction` carries `gate_verdict`. Sending `gated=true` to a
  // kind with no gate would ask for a condition nothing can satisfy and blank
  // the section — so the flag is per-kind, and these three opt out until
  // their own predictions grow a gate.
  it.each(["species", "headtail", "dive-slate"])(
    "does not send gated for %s, which has no gate",
    async (kind) => {
      const fetchMock = vi.fn<FetchSig>(async () => jsonResponse([]));
      vi.stubGlobal("fetch", fetchMock);

      await getIncompleteProjectIds(60);

      const call = fetchMock.mock.calls.find(([url]) =>
        url.includes(`/labels/${kind}/`),
      );
      expect(call?.[0]).not.toContain("gated");
    },
  );

  it("still asks for incomplete work alongside the gate filter", async () => {
    const fetchMock = vi.fn<FetchSig>(async () => jsonResponse([]));
    vi.stubGlobal("fetch", fetchMock);

    await getIncompleteProjectIds(60);

    for (const [url] of fetchMock.mock.calls) {
      expect(url).toContain("incomplete=true");
    }
  });
});
