# fishsense-lite-web

Next.js 15 (App Router) + React + TypeScript + Tailwind web app at
`fishsense.e4e.ucsd.edu`. Two surfaces:

- **Public landing (`/`)** — SSR fetches active project IDs from
  `fishsense-api`, resolves names from Label Studio, renders
  categorized link cards (results dashboards + admin tools). Replaces
  the prior mafl dashboard + its hourly config-writer workflow.
- **Authenticated portal (`/portal/*`)** — gated by
  [Auth.js](https://authjs.dev/) v5 (`next-auth`) with Authentik as the
  OIDC provider. App-owned JWT session; future per-user content goes
  here.

The landing is intentionally decoupled from `AUTH_*` env so it keeps
rendering even before the OIDC client is set up.

## Layout

```
app/
  page.tsx                       # public landing
  layout.tsx                     # root layout (Tailwind)
  portal/page.tsx                # authenticated portal (server component)
  api/auth/[...nextauth]/route.ts # NextAuth handlers
auth.ts                          # NextAuth config (function-form, lazy env)
middleware.ts                    # gates /portal/:path*
lib/
  env.ts                         # strict env-var proxy (fails loudly)
  active-projects.ts             # SSR fetch of LS-project IDs
  fishsense-api.ts, label-studio.ts
  sections.ts, static-links.ts
  auth-callbacks.ts              # pure jwt/session callbacks
types/next-auth.d.ts             # session.accessToken + session.user.groups
```

## Required env

All env vars are required (the `env` proxy in [lib/env.ts](lib/env.ts)
throws on first access if any are missing). Canonical shape lives in
[.env.example](.env.example); on prod hosts the file is
`web_volumes/.env` (untracked, populated by ops).

| Var | Purpose |
|---|---|
| `FISHSENSE_API_URL` | Base URL for fishsense-api. Inside the prod docker network this is `http://fishsense-api:8000` (skips Traefik + Authentik). |
| `FISHSENSE_API_USERNAME`, `FISHSENSE_API_PASSWORD` | Service-account basic auth for SSR fetches. |
| `LABEL_STUDIO_URL`, `LABEL_STUDIO_API_KEY` | Label Studio access for project-name resolution. |
| `AUTH_SECRET` | NextAuth JWT signing secret. Generate with `openssl rand -base64 32`. |
| `AUTH_AUTHENTIK_ID`, `AUTH_AUTHENTIK_SECRET` | OIDC client credentials issued by Authentik. |
| `AUTH_AUTHENTIK_ISSUER` | Authentik issuer URL **including the application slug**, no trailing slash, e.g. `https://auth.e4e.ucsd.edu/application/o/fishsense-lite-web`. |

## Deploying

This is a workspace member of the prod orchestrator stack
(`deploy/compose.yml`). Deploy is automated:

1. Land changes on `main` → `build.yml` ships
   `ghcr.io/ucsd-e4e/fishsense-lite-web:sha-<short>`.
2. release-please opens a release PR; merging it triggers `promote.yml`
   which retags to `:v<version>` and opens an `auto-deploy/*` PR
   bumping the image pin in `deploy/compose.yml`.
3. Merging the auto-deploy PR triggers `deploy.yml` on the
   `fishsense-prod` self-hosted runner, which `git pull`s + `docker
   compose pull && up -d`s the orchestrator host.

See [../../README.md](../../README.md) and
[../../deploy/README.md](../../deploy/README.md) for the full pipeline.

### One-time configuration (Authentik OIDC)

Before the portal can authenticate users:

1. In the Authentik admin UI, create an **OAuth2/OpenID Provider**
   bound to a signing key, with scopes `openid profile email` (and
   `groups` if the default `profile` claim doesn't already include
   them). Add the redirect URI:
   ```
   https://fishsense.e4e.ucsd.edu/api/auth/callback/authentik
   ```
2. Create an **Application** bound to that provider (slug
   `fishsense-lite-web`). The issuer URL it exposes is what goes into
   `AUTH_AUTHENTIK_ISSUER` (with the slug, no trailing slash).
3. Add the four `AUTH_*` keys to `web_volumes/.env` on the orchestrator
   host. Generate `AUTH_SECRET` with `openssl rand -base64 32`.
4. Restart the container:
   `docker compose -f deploy/compose.yml up -d fishsense-lite-web`.
5. Smoke-test: hit `/`, click **Sign in**, complete the Authentik
   flow, land on `/portal` showing your name/email/groups.

Group-based authorization is intentionally not enforced yet. When you
need it, gate inside [auth.ts](auth.ts)
(`callbacks.authorized`) or check `session.user.groups` server-side
in `app/portal/*` — groups are already plumbed onto the session.

## Local development

The repo's devcontainer (`deploy/compose.local.yml`) does not boot
this app — it's a Node service while the local stack is Python +
Postgres + Temporal + Label Studio. Run it manually:

```
cd apps/fishsense-lite-web
npm install
cp .env.example .env.local      # then fill in real values
npm run dev                      # next dev on :3000
```

For local dev without an Authentik tenant available you can either
(a) leave `AUTH_*` unset and avoid `/portal` (the landing still works
because env reads are lazy), or (b) point at a dev Authentik instance.

### Common scripts

```
npm run dev               # next dev (hot reload)
npm run build             # next build (production)
npm start                 # serve the standalone build
npm run lint              # next lint (ESLint)
npm run typecheck         # tsc --noEmit
npm test                  # vitest run (unit tests in lib/)
npm run test:integration  # vitest with the integration config
```

## Tests

Unit tests live next to their modules in `lib/*.test.ts` and run
under vitest. The default config (`vitest.config.ts`) only includes
`lib/**/*.test.ts`; the integration config
(`vitest.config.integration.ts`) targets `tests/integration/*`.

Per the repo's [CLAUDE.md](../../CLAUDE.md), TDD is mandatory for
non-trivial logic — write a failing test first. UI rendering is the
narrow exception; extract logic into a unit and cover it there.

## Production caveats

- `fishsense.e4e.ucsd.edu` (this app) is **not** behind the
  Authentik forward-auth Traefik middleware that fronts
  `orchestrator.fishsense.e4e.ucsd.edu` (fishsense-api, qcomm static
  server). Auth is app-owned and only gates `/portal` (the SSR check
  lives in `app/portal/page.tsx`); the landing must stay public.
- SSR fetches use the in-cluster URL `http://fishsense-api:8000` to
  bypass the public Traefik route + Authentik middleware that 302s
  basic-auth. Don't switch this to the public hostname.
- `AUTH_URL` is **strongly recommended** in any non-localhost deploy.
  NextAuth v5 + `trustHost: true` is *supposed* to derive the base
  URL from the inbound `Host`/`X-Forwarded-Host` header, but in
  practice the providers/picker route and the OAuth post-sign-in
  redirect can still end up built against the container's internal
  listen address (`http://0.0.0.0:3000/...`) — which 500s the OAuth
  callback and sends the browser somewhere unreachable. Set
  `AUTH_URL=https://fishsense.e4e.ucsd.edu` to bypass header
  detection entirely. See `deploy/web_volumes/.env.example` and the
  orchestrator-bootstrap section of the repo's `CLAUDE.md`.
