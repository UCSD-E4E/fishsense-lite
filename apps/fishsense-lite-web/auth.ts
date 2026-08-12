import NextAuth from "next-auth";
import Authentik from "next-auth/providers/authentik";
import { jwtCallback, sessionCallback } from "@/lib/auth-callbacks";
import { env } from "@/lib/env";

// Function-form config: env is read per-request, not at module load.
// `next build` imports this module to collect page data without AUTH_*
// env vars set, so reading env eagerly here would fail the build.
export const { auth, handlers, signIn, signOut } = NextAuth(() => ({
  secret: env.authSecret,
  trustHost: true,
  session: { strategy: "jwt" },
  providers: [
    Authentik({
      clientId: env.authAuthentikId,
      clientSecret: env.authAuthentikSecret,
      issuer: env.authAuthentikIssuer,
      // `groups` must be requested explicitly — the provider's default scope
      // is `openid profile email`, so without this the userinfo response
      // carries no groups, `session.user.groups` is always `[]`, and the
      // portal's authorization check would deny everyone. This is why the
      // portal used to render "no groups" for every signed-in user.
      //
      // Requires a matching scope + property mapping on the FishSense OIDC
      // application in Authentik, exactly as the Superset app needs (see
      // `superset_config.py`'s `'scope': 'openid email profile groups'`).
      authorization: { params: { scope: "openid email profile groups" } },
    }),
  ],
  callbacks: {
    jwt: jwtCallback,
    session: sessionCallback,
  },
}));
