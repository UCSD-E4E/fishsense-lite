import { describe, expect, it } from "vitest";

const WEB_URL = process.env.FISHSENSE_WEB_URL ?? "http://localhost:3000";

describe("/portal SSR auth gate (against the local stack)", () => {
  it("redirects signed-out GET /portal to the next-auth sign-in route", async () => {
    // Pins the workaround for the Next.js 15.5 middleware-loader bug —
    // the auth gate now lives in app/portal/page.tsx instead of
    // middleware.ts, so a regression that drops the redirect (or one
    // that re-introduces a broken edge middleware) would surface here
    // as a 200 / 500 instead of a 307.
    const res = await fetch(`${WEB_URL}/portal`, {
      cache: "no-store",
      redirect: "manual",
    });
    expect(res.status).toBe(307);
    const location = res.headers.get("location") ?? "";
    expect(location).toContain("/api/auth/signin");
    expect(location).toContain("callbackUrl=%2Fportal");
  });
});
