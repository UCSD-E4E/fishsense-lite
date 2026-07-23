import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  clearCalibrationSource,
  getDives,
  setCalibrationSource,
} from "./dives";

beforeEach(() => {
  vi.stubEnv("FISHSENSE_API_URL", "http://api.test");
  vi.stubEnv("FISHSENSE_API_USERNAME", "alice");
  vi.stubEnv("FISHSENSE_API_PASSWORD", "secret");
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

const EXPECTED_AUTH = `Basic ${Buffer.from("alice:secret").toString("base64")}`;

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    statusText: init.statusText ?? "OK",
    headers: { "content-type": "application/json", ...(init.headers ?? {}) },
  });
}

type NextFetchInit = RequestInit & { next?: { revalidate?: number } };
type FetchSig = (input: string, init?: NextFetchInit) => Promise<Response>;

describe("getDives", () => {
  it("GETs the dives list with Basic auth and returns the parsed body", async () => {
    const dives = [
      { id: 1, name: "slate dive", dive_slate_id: 5, calibration_dive_id: null },
      { id: 2, name: "fish dive", dive_slate_id: null, calibration_dive_id: 1 },
    ];
    const fetchMock = vi.fn<FetchSig>(async () => jsonResponse(dives));
    vi.stubGlobal("fetch", fetchMock);

    const result = await getDives(30);

    expect(result).toEqual(dives);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://api.test/api/v1/dives/");
    const headers = init?.headers as Record<string, string>;
    expect(headers.Authorization).toBe(EXPECTED_AUTH);
    expect(init?.next?.revalidate).toBe(30);
  });

  it("throws on a non-OK response", async () => {
    const fetchMock = vi.fn<FetchSig>(async () =>
      new Response("nope", { status: 500, statusText: "Server Error" }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(getDives()).rejects.toThrow(/dives list failed: 500/);
  });
});

describe("setCalibrationSource", () => {
  it("PUTs to the path-param endpoint with Basic auth", async () => {
    const fetchMock = vi.fn<FetchSig>(async () => jsonResponse(2));
    vi.stubGlobal("fetch", fetchMock);

    await setCalibrationSource(2, 1);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://api.test/api/v1/dives/2/calibration-source/1");
    expect(init?.method).toBe("PUT");
    const headers = init?.headers as Record<string, string>;
    expect(headers.Authorization).toBe(EXPECTED_AUTH);
  });

  it("throws on a non-OK response", async () => {
    const fetchMock = vi.fn<FetchSig>(async () =>
      new Response("bad", { status: 400, statusText: "Bad Request" }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(setCalibrationSource(1, 1)).rejects.toThrow(
      /set calibration source failed: 400/,
    );
  });
});

describe("clearCalibrationSource", () => {
  it("DELETEs the calibration-source endpoint with Basic auth", async () => {
    const fetchMock = vi.fn<FetchSig>(async () => new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    await clearCalibrationSource(2);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://api.test/api/v1/dives/2/calibration-source/");
    expect(init?.method).toBe("DELETE");
    const headers = init?.headers as Record<string, string>;
    expect(headers.Authorization).toBe(EXPECTED_AUTH);
  });

  it("throws on a non-OK response", async () => {
    const fetchMock = vi.fn<FetchSig>(async () =>
      new Response("nope", { status: 404, statusText: "Not Found" }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(clearCalibrationSource(9)).rejects.toThrow(
      /clear calibration source failed: 404/,
    );
  });
});
