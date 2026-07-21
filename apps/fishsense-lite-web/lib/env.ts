function required(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return value;
}

const ENV_VARS = {
  fishsenseApiUrl: "FISHSENSE_API_URL",
  fishsenseApiUsername: "FISHSENSE_API_USERNAME",
  fishsenseApiPassword: "FISHSENSE_API_PASSWORD",
  labelStudioUrl: "LABEL_STUDIO_URL",
  labelStudioApiKey: "LABEL_STUDIO_API_KEY",
  authSecret: "AUTH_SECRET",
  authAuthentikId: "AUTH_AUTHENTIK_ID",
  authAuthentikSecret: "AUTH_AUTHENTIK_SECRET",
  authAuthentikIssuer: "AUTH_AUTHENTIK_ISSUER",
} as const;

type EnvKey = keyof typeof ENV_VARS;
type Env = Record<EnvKey, string>;

export const env: Env = new Proxy({} as Env, {
  get(_target, prop) {
    if (typeof prop === "string" && prop in ENV_VARS) {
      return required(ENV_VARS[prop as EnvKey]);
    }
    return undefined;
  },
});

const TRUTHY = new Set(["true", "1", "yes"]);

// Kill-switch for the Label Studio integration. **Defaults to off**, but
// both original blockers are fixed as of 2026-07-21, so prod sets
// `LABEL_STUDIO_ENABLED=true` (see deploy/incus/compose.yml).
//
// What used to break:
//   1. Auth — this app sent `Authorization: Token <key>` and every
//      `/api/projects/<id>` fetch 401'd. The hosted instance
//      (app.heartex.com) treats `LABEL_STUDIO_API_KEY` as a *refresh*
//      token: it must be exchanged at `/api/token/refresh` for a
//      short-lived access JWT sent as `Bearer`. `lib/label-studio.ts`
//      now does that (and caches/dedupes the exchange).
//   2. Blast radius — `getProject` throws on non-OK and `getProjects`
//      fanned out under `Promise.all`, so one dead ID rejected out of the
//      server component and 500'd the whole landing page. fishsense-api
//      still stores legacy ids (57-117) that all 404 on the hosted
//      instance, so that was guaranteed. `getProjects` now uses
//      `Promise.allSettled` and drops what it can't resolve.
//
// Kept as a switch so the integration can still be cut fast if the hosted
// instance misbehaves. Disabled, `getActiveProjects` returns empty buckets
// and `buildSections` collapses the four labeling sections.
//
// Compared explicitly against a truthy allowlist rather than
// `Boolean(process.env.X)` — the latter reads the literal string "false"
// as true. Same footgun as `E4EFS_DOCKER`; see CLAUDE.md.
export function labelStudioEnabled(): boolean {
  const value = process.env.LABEL_STUDIO_ENABLED;
  return value !== undefined && TRUTHY.has(value.toLowerCase());
}

export const __test = { required };
