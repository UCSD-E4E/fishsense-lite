import { describe, expect, it } from "vitest";
import { jwtCallback, sessionCallback } from "./auth-callbacks";

const baseToken = { sub: "u1", name: "User One", email: "u@e.com" };

describe("jwtCallback", () => {
  it("copies access_token from account on initial sign-in", async () => {
    const result = await jwtCallback({
      token: { ...baseToken },
      account: { access_token: "at-123", provider: "authentik" } as never,
      profile: { groups: ["a", "b"] } as never,
    });
    expect(result.accessToken).toBe("at-123");
  });

  it("copies groups from profile on initial sign-in", async () => {
    const result = await jwtCallback({
      token: { ...baseToken },
      account: { access_token: "at-1" } as never,
      profile: { groups: ["fishsense-admins", "labelers"] } as never,
    });
    expect(result.groups).toEqual(["fishsense-admins", "labelers"]);
  });

  it("defaults groups to [] when profile has none", async () => {
    const result = await jwtCallback({
      token: { ...baseToken },
      account: { access_token: "at-1" } as never,
      profile: {} as never,
    });
    expect(result.groups).toEqual([]);
  });

  it("returns token unchanged on subsequent calls (no account)", async () => {
    const existing = { ...baseToken, accessToken: "at-prev", groups: ["x"] };
    const result = await jwtCallback({
      token: existing,
      account: null,
      profile: undefined,
    });
    expect(result.accessToken).toBe("at-prev");
    expect(result.groups).toEqual(["x"]);
  });
});

describe("sessionCallback", () => {
  it("surfaces accessToken and groups from token onto session", async () => {
    const result = await sessionCallback({
      session: { user: { name: "User One", email: "u@e.com" }, expires: "2099-01-01" } as never,
      token: { ...baseToken, accessToken: "at-9", groups: ["g1"] } as never,
    });
    expect(result.accessToken).toBe("at-9");
    expect(result.user.groups).toEqual(["g1"]);
  });

  it("defaults groups to [] when token has none", async () => {
    const result = await sessionCallback({
      session: { user: { name: "U", email: "u@e.com" }, expires: "2099-01-01" } as never,
      token: { ...baseToken } as never,
    });
    expect(result.user.groups).toEqual([]);
    expect(result.accessToken).toBeUndefined();
  });
});
