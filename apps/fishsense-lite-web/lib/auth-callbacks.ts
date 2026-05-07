import type { Account, Profile, Session, User } from "next-auth";
import type { JWT } from "next-auth/jwt";

interface AuthentikProfileLike extends Profile {
  groups?: string[];
}

interface JwtCallbackArgs {
  token: JWT;
  account?: Account | null;
  profile?: AuthentikProfileLike;
  user?: User;
}

export async function jwtCallback({
  token,
  account,
  profile,
  user,
}: JwtCallbackArgs): Promise<JWT> {
  if (account) {
    // Temporary diagnostic for the empty /portal user-info problem.
    // Remove once the OIDC profile shape is confirmed in prod logs.
    console.log("[auth] jwtCallback initial sign-in", {
      account: { provider: account.provider, type: account.type, hasAccessToken: typeof account.access_token === "string" },
      user,
      profileKeys: profile ? Object.keys(profile) : null,
      profile,
    });
    if (typeof account.access_token === "string") {
      token.accessToken = account.access_token;
    }
    token.groups = Array.isArray(profile?.groups) ? profile.groups : [];
    if (user) {
      if (typeof user.id === "string") token.sub = user.id;
      if (typeof user.name === "string") token.name = user.name;
      if (typeof user.email === "string") token.email = user.email;
      if (typeof user.image === "string") token.picture = user.image;
    }
  }
  return token;
}

interface SessionCallbackArgs {
  session: Session;
  token: JWT;
}

export async function sessionCallback({ session, token }: SessionCallbackArgs): Promise<Session> {
  if (typeof token.accessToken === "string") {
    session.accessToken = token.accessToken;
  }
  if (typeof token.sub === "string") session.user.id = token.sub;
  if (typeof token.name === "string") session.user.name = token.name;
  if (typeof token.email === "string") session.user.email = token.email;
  if (typeof token.picture === "string") session.user.image = token.picture;
  session.user.groups = Array.isArray(token.groups) ? token.groups : [];
  return session;
}
