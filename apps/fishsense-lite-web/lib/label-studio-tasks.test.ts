import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { __resetTokenCache } from "./label-studio";
import { acceptPrediction, fetchTaskImage, listTasks } from "./label-studio-tasks";

beforeEach(() => {
  vi.stubEnv("LABEL_STUDIO_URL", "http://ls.test");
  vi.stubEnv("LABEL_STUDIO_API_KEY", "pat");
  __resetTokenCache();
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

/** Answers the token refresh, then delegates everything else to `handler`. */
function mockFetch(handler: (url: string, init?: RequestInit) => Promise<Response>) {
  const fn = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith("/api/token/refresh")) return json({ access: "jwt" });
    return handler(url, init);
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

describe("listTasks", () => {
  it("returns the page's tasks", async () => {
    mockFetch(async () => json({ tasks: [{ id: 1 }, { id: 2 }], total: 2 }));
    const page = await listTasks(9, 1);
    expect(page.tasks.map((t) => t.id)).toEqual([1, 2]);
    expect(page.total).toBe(2);
  });

  // DRF answers a page past the end of a result set with 404
  // `{"detail": "Invalid page."}` rather than an empty list. Treating that as
  // a failure surfaced a fatal error mid-scan in the Android app.
  it("reads an out-of-range page as drained, not as an error", async () => {
    mockFetch(async () => json({ detail: "Invalid page." }, 404));
    const page = await listTasks(9, 7);
    expect(page.tasks).toEqual([]);
    expect(page.drained).toBe(true);
  });

  it("still throws on a real server error", async () => {
    mockFetch(async () => new Response("boom", { status: 500 }));
    await expect(listTasks(9, 1)).rejects.toThrow(/500/);
  });

  it("refreshes the access token once on 401 and retries", async () => {
    let calls = 0;
    const fetchMock = mockFetch(async () => {
      calls += 1;
      return calls === 1 ? json({ detail: "unauthorised" }, 401) : json({ tasks: [{ id: 3 }] });
    });
    const page = await listTasks(9, 1);
    expect(page.tasks.map((t) => t.id)).toEqual([3]);
    // refresh, 401, refresh, success
    expect(fetchMock.mock.calls.filter(([u]) => u.endsWith("/api/token/refresh"))).toHaveLength(2);
  });

  it("accepts a bare array body", async () => {
    // Some Label Studio versions return a list rather than {tasks, total}.
    mockFetch(async () => json([{ id: 4 }]));
    const page = await listTasks(9, 1);
    expect(page.tasks.map((t) => t.id)).toEqual([4]);
  });
});

describe("acceptPrediction", () => {
  it("posts the prediction result verbatim", async () => {
    let sent: unknown = null;
    mockFetch(async (url, init) => {
      if (url.includes("/annotations/")) {
        sent = JSON.parse(String(init?.body));
        return json({ id: 555 });
      }
      throw new Error(`unexpected ${url}`);
    });

    const result = [
      {
        from_name: "laser",
        to_name: "img",
        type: "keypointlabels",
        original_width: 4000,
        original_height: 3000,
        image_rotation: 0,
        value: { x: 57.925, y: 46.966, keypointlabels: ["Red Laser"] },
      },
    ];

    const id = await acceptPrediction(42, result, 3200);
    expect(id).toBe(555);

    const body = sent as { result: unknown; was_cancelled?: boolean; lead_time?: number };
    // Byte-for-byte: this equality IS the safety argument for accepting.
    expect(body.result).toEqual(result);
    expect(body.lead_time).toBe(3200);
    // A cancelled annotation would flip `completed` with no coordinates.
    expect(body.was_cancelled).toBeFalsy();
  });

  it("throws when Label Studio rejects the annotation", async () => {
    mockFetch(async () => new Response("nope", { status: 400 }));
    await expect(acceptPrediction(42, [], 10)).rejects.toThrow(/400/);
  });
});

describe("fetchTaskImage", () => {
  // `resolve_uri` does NOT return a presigned S3 URL — it returns a path on
  // Label Studio's own API server, and that path is authenticated. Getting the
  // shape wrong reported "queue empty" against a project holding 283 tasks.
  it("hits the resolve path with the base64 of the s3 URI", async () => {
    let seen = "";
    mockFetch(async (url) => {
      seen = url;
      return new Response("jpegbytes", { status: 200 });
    });

    await fetchTaskImage(42, "s3://bucket/preprocess_jpeg/abc.JPG");

    expect(seen).toContain("/tasks/42/resolve/?fileuri=");
    const encoded = new URL(seen).searchParams.get("fileuri") ?? "";
    expect(Buffer.from(encoded, "base64").toString("utf8")).toBe(
      "s3://bucket/preprocess_jpeg/abc.JPG",
    );
  });
});

describe("rate limiting", () => {
  // Hosted Label Studio 429s a burst. Surfacing it aborts the whole queue load
  // over a condition that clears by itself — which is what took the triage page
  // down on its first real run.
  it("retries a 429 and returns the eventual success", async () => {
    let calls = 0;
    mockFetch(async () => {
      calls += 1;
      return calls === 1
        ? new Response("slow down", { status: 429, headers: { "retry-after": "0" } })
        : json({ tasks: [{ id: 7 }] });
    });

    const page = await listTasks(9, 1);
    expect(page.tasks.map((t) => t.id)).toEqual([7]);
    expect(calls).toBe(2);
  });

  it("gives up after a bounded number of retries rather than hanging", async () => {
    let calls = 0;
    mockFetch(async () => {
      calls += 1;
      return new Response("slow down", { status: 429, headers: { "retry-after": "0" } });
    });

    await expect(listTasks(9, 1)).rejects.toThrow(/429/);
    // First attempt plus a bounded number of retries — never unbounded.
    expect(calls).toBeGreaterThan(1);
    expect(calls).toBeLessThanOrEqual(5);
  });
});
