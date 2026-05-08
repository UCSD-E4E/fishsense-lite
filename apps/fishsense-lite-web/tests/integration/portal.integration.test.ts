import { encode } from "next-auth/jwt";
import { describe, expect, it } from "vitest";

const WEB_URL = process.env.FISHSENSE_WEB_URL ?? "http://localhost:3000";

// Must match the AUTH_SECRET set on the fishsense-lite-web service in
// deploy/compose.local.yml — the page's auth() call decrypts the
// session cookie with whatever secret the container booted with, so
// drift here would produce a 307 redirect instead of a 200 (next-auth
// silently treats an undecryptable cookie as signed-out).
const AUTH_SECRET = "fishsense_local_test_auth_secret_only_for_ci_v1";

// Local stack runs over http, so the cookie name has no `__Secure-`
// prefix. JWE encryption salt defaults to the cookie name in Auth.js v5.
const SESSION_COOKIE_NAME = "authjs.session-token";

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

  it("renders the user-info page when a valid session cookie is presented", async () => {
    // Mints a session JWE the same way next-auth does on a real OIDC
    // callback, then asserts the page reads it and renders the user
    // dl. Skips the OAuth dance entirely — what's covered is the
    // session.user → DOM contract (lib/auth-callbacks.ts plumbs token
    // → session, app/portal/page.tsx renders session.user). A change
    // that drops a field from the dl, breaks the cookie name, or
    // mis-salts the JWE would surface here.
    const sessionToken = await encode({
      token: {
        sub: "test-user-id",
        name: "Integration Test User",
        email: "integration-test@fishsense.local",
        groups: ["test-group"],
      },
      secret: AUTH_SECRET,
      salt: SESSION_COOKIE_NAME,
      maxAge: 60 * 60,
    });

    const res = await fetch(`${WEB_URL}/portal`, {
      cache: "no-store",
      redirect: "manual",
      headers: { cookie: `${SESSION_COOKIE_NAME}=${sessionToken}` },
    });
    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).toContain("Integration Test User");
    expect(body).toContain("integration-test@fishsense.local");
    expect(body).toContain("test-group");
  });
});
