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
    }),
  ],
  callbacks: {
    jwt: jwtCallback,
    session: sessionCallback,
  },
}));
