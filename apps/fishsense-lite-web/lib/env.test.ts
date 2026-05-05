import { afterEach, describe, expect, it, vi } from "vitest";
import { __test, env } from "./env";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("required", () => {
  it("returns the env var when set", () => {
    vi.stubEnv("FOO", "bar");
    expect(__test.required("FOO")).toBe("bar");
  });

  it("throws when missing", () => {
    vi.stubEnv("FOO", "");
    expect(() => __test.required("FOO")).toThrow(/Missing required env var: FOO/);
  });

  it("throws when undefined", () => {
    expect(() => __test.required("DEFINITELY_NOT_SET_X9Q1")).toThrow();
  });
});

describe("env proxy", () => {
  it("reads each declared key lazily from process.env", () => {
    vi.stubEnv("FISHSENSE_API_URL", "http://api");
    vi.stubEnv("FISHSENSE_API_USERNAME", "u");
    vi.stubEnv("FISHSENSE_API_PASSWORD", "p");
    vi.stubEnv("LABEL_STUDIO_URL", "http://ls");
    vi.stubEnv("LABEL_STUDIO_API_KEY", "k");

    expect(env.fishsenseApiUrl).toBe("http://api");
    expect(env.fishsenseApiUsername).toBe("u");
    expect(env.fishsenseApiPassword).toBe("p");
    expect(env.labelStudioUrl).toBe("http://ls");
    expect(env.labelStudioApiKey).toBe("k");
  });

  it("throws on access when an env var is missing", () => {
    vi.stubEnv("FISHSENSE_API_URL", "");
    expect(() => env.fishsenseApiUrl).toThrow(/FISHSENSE_API_URL/);
  });

  it("re-reads on each access (no stale cache)", () => {
    vi.stubEnv("FISHSENSE_API_URL", "http://first");
    expect(env.fishsenseApiUrl).toBe("http://first");
    vi.stubEnv("FISHSENSE_API_URL", "http://second");
    expect(env.fishsenseApiUrl).toBe("http://second");
  });
});
