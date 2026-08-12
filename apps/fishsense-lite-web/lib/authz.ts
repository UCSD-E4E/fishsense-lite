import type { Session } from "next-auth";

/**
 * Authorization for `/portal`.
 *
 * Signing in proves only that someone holds an account in the Authentik
 * realm — which is the whole university SSO population, not the FishSense
 * team. Before this existed the portal checked authentication and nothing
 * else, so any realm account could rewrite `calibration_dive_id` on any dive.
 * That is a live pipeline input: a dive measured off a borrowed calibration
 * runs -8..+2% error against ~1% for its own, so the link changes reported
 * fish lengths.
 *
 * `session.user.groups` was already plumbed profile -> JWT -> session, and was
 * only ever rendered as text. This is the missing consumer.
 */

/** Groups permitted to use the portal, from `PORTAL_ALLOWED_GROUPS`. */
export function portalAllowedGroups(): string[] {
  return (process.env.PORTAL_ALLOWED_GROUPS ?? "")
    .split(",")
    .map((group) => group.trim())
    .filter(Boolean);
}

/**
 * True iff `session` belongs to a signed-in user in an allowed group.
 *
 * **Fails closed.** An unset or empty `PORTAL_ALLOWED_GROUPS` denies
 * everyone rather than admitting everyone: a check that defaults to open is
 * not a check, and a misconfigured deploy should lock the portal, not
 * silently reopen the hole. This mirrors the Superset posture already running
 * in this deployment — a user in none of the mapped groups gets Public, i.e.
 * no access (`superset_config.AUTH_ROLES_MAPPING`).
 *
 * Group names are matched exactly. Authentik's are case-sensitive, and a
 * loose match would make the allowlist advisory.
 */
export function isPortalAuthorized(session: Session | null): boolean {
  if (!session?.user) return false;
  const allowed = portalAllowedGroups();
  if (allowed.length === 0) return false;
  const groups = session.user.groups ?? [];
  return groups.some((group) => allowed.includes(group));
}
