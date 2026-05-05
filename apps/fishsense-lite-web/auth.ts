import NextAuth from "next-auth";
import Authentik from "next-auth/providers/authentik";
import { jwtCallback, sessionCallback } from "@/lib/auth-callbacks";
import { env } from "@/lib/env";

export const { auth, handlers, signIn, signOut } = NextAuth({
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
});
