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

// Kill-switch for the Label Studio integration. **Defaults to off.**
//
// The hosted instance (app.heartex.com) rejects the legacy
// `Authorization: Token <key>` scheme this app sends, so every
// `/api/projects/<id>` fetch 401s. `getProject` throws on a non-OK
// response and `getProjects` fans them out through `Promise.all`, so a
// single bad ID rejected out of the server component and 500'd the whole
// landing page. Disabled, `getActiveProjects` returns empty buckets and
// `buildSections` collapses the four labeling sections.
//
// Re-enable with `LABEL_STUDIO_ENABLED=true` once the new instance's auth
// scheme and project IDs are reconciled with what fishsense-api stores.
//
// Compared explicitly against a truthy allowlist rather than
// `Boolean(process.env.X)` — the latter reads the literal string "false"
// as true. Same footgun as `E4EFS_DOCKER`; see CLAUDE.md.
export function labelStudioEnabled(): boolean {
  const value = process.env.LABEL_STUDIO_ENABLED;
  return value !== undefined && TRUTHY.has(value.toLowerCase());
}

export const __test = { required };
