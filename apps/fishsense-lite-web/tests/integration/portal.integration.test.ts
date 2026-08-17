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

// Must match `PORTAL_ALLOWED_GROUPS` on the fishsense-lite-web service in
// deploy/compose.local.yml. `isPortalAuthorized` fails closed, so an unset
// allowlist denies everyone — a drift here renders the not-authorized dead
// end instead of the portal, which is exactly how this test first failed.
const ALLOWED_GROUP = "test-group";

/** Mint a session JWE the way next-auth does on a real OIDC callback. */
async function sessionCookie(groups: string[]) {
  return encode({
    token: {
      sub: "test-user-id",
      name: "Integration Test User",
      email: "integration-test@fishsense.local",
      groups,
    },
    secret: AUTH_SECRET,
    salt: SESSION_COOKIE_NAME,
    maxAge: 60 * 60,
  });
}

async function getPortal(cookie?: string) {
  return fetch(`${WEB_URL}/portal`, {
    cache: "no-store",
    redirect: "manual",
    ...(cookie
      ? { headers: { cookie: `${SESSION_COOKIE_NAME}=${cookie}` } }
      : {}),
  });
}

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
    const res = await getPortal(await sessionCookie([ALLOWED_GROUP]));

    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).toContain("Integration Test User");
    expect(body).toContain("integration-test@fishsense.local");
    expect(body).toContain(ALLOWED_GROUP);
  });

  it("renders a dead end, not the portal, for a signed-in user in no allowed group", async () => {
    // The authorization half of the gate, which nothing covered end to end.
    // Signing in only proves an account in the Authentik realm — the whole
    // university SSO population, not the FishSense team. Before the gate
    // existed, any realm account could rewrite `calibration_dive_id`, which
    // is a live pipeline input: a dive measured off a borrowed calibration
    // runs -8..+2% error against ~1% for its own.
    //
    // Asserted as an absence as well as a presence, because the failure that
    // matters is the gate silently disappearing — a test that only checked
    // for the denial text would still pass if the page rendered BOTH.
    const res = await getPortal(await sessionCookie(["some-other-group"]));

    expect(res.status).toBe(200);
    const body = await res.text();
    expect(body).toContain("not in a group permitted");
    expect(body).not.toContain("Dive calibration links");
  });

  it("does not redirect an authorized-but-signed-in user into a sign-in loop", async () => {
    // A denied user is already signed in, so redirecting them to sign-in
    // would loop forever. The dead end has to be a 200 with a sign-out
    // affordance instead — pinned here because "redirect on failure" is the
    // reflexive fix someone will reach for.
    const res = await getPortal(await sessionCookie([]));

    expect(res.status).toBe(200);
    expect(res.headers.get("location")).toBeNull();
  });
});
