import { afterEach, describe, expect, it } from "vitest";
import type { Session } from "next-auth";
import { isPortalAuthorized, portalAllowedGroups } from "./authz";

const ORIGINAL = process.env.PORTAL_ALLOWED_GROUPS;

afterEach(() => {
  if (ORIGINAL === undefined) delete process.env.PORTAL_ALLOWED_GROUPS;
  else process.env.PORTAL_ALLOWED_GROUPS = ORIGINAL;
});

function session(groups: string[] | undefined): Session {
  return { user: { groups }, expires: "" } as unknown as Session;
}

describe("portalAllowedGroups", () => {
  it("parses a comma-separated list", () => {
    process.env.PORTAL_ALLOWED_GROUPS = "fishsense-portal-admin,fishsense-portal-editor";
    expect(portalAllowedGroups()).toEqual([
      "fishsense-portal-admin",
      "fishsense-portal-editor",
    ]);
  });

  it("trims whitespace and drops empty entries", () => {
    process.env.PORTAL_ALLOWED_GROUPS = " a , ,b, ";
    expect(portalAllowedGroups()).toEqual(["a", "b"]);
  });

  it("is empty when unset", () => {
    delete process.env.PORTAL_ALLOWED_GROUPS;
    expect(portalAllowedGroups()).toEqual([]);
  });
});

describe("isPortalAuthorized", () => {
  it("denies when PORTAL_ALLOWED_GROUPS is unset", () => {
    // Fail CLOSED. An authorization check that defaults to open is not an
    // authorization check — and the portal writes `calibration_dive_id`,
    // which changes measured fish lengths. Matches Superset's posture: a
    // user in none of the mapped groups gets no access.
    delete process.env.PORTAL_ALLOWED_GROUPS;
    expect(isPortalAuthorized(session(["fishsense-portal-admin"]))).toBe(false);
  });

  it("denies when the allowlist is present but empty", () => {
    process.env.PORTAL_ALLOWED_GROUPS = "   ";
    expect(isPortalAuthorized(session(["fishsense-portal-admin"]))).toBe(false);
  });

  it("allows a user in an allowed group", () => {
    process.env.PORTAL_ALLOWED_GROUPS = "fishsense-portal-admin";
    expect(isPortalAuthorized(session(["other", "fishsense-portal-admin"]))).toBe(true);
  });

  it("denies an authenticated user in no allowed group", () => {
    // The whole point: authentication is realm-wide, so every Authentik
    // account could reach the portal before this existed.
    process.env.PORTAL_ALLOWED_GROUPS = "fishsense-portal-admin";
    expect(isPortalAuthorized(session(["some-other-team"]))).toBe(false);
  });

  it("denies when the token carried no groups at all", () => {
    // The `groups` OIDC scope has to be requested AND mapped on the
    // Authentik application. If either is missing every session arrives with
    // no groups, and denying is the correct response.
    process.env.PORTAL_ALLOWED_GROUPS = "fishsense-portal-admin";
    expect(isPortalAuthorized(session(undefined))).toBe(false);
    expect(isPortalAuthorized(session([]))).toBe(false);
  });

  it("denies when there is no session or no user", () => {
    process.env.PORTAL_ALLOWED_GROUPS = "fishsense-portal-admin";
    expect(isPortalAuthorized(null)).toBe(false);
    expect(isPortalAuthorized({ expires: "" } as unknown as Session)).toBe(false);
  });

  it("matches group names exactly", () => {
    // Authentik group names are case-sensitive; a loose match would let
    // "Fishsense-Portal-Admin" through and make the allowlist advisory.
    process.env.PORTAL_ALLOWED_GROUPS = "fishsense-portal-admin";
    expect(isPortalAuthorized(session(["Fishsense-Portal-Admin"]))).toBe(false);
    expect(isPortalAuthorized(session(["fishsense-portal-admin-extra"]))).toBe(false);
  });
});
