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
  // Temporary diagnostic for the empty /portal user-info problem.
  // Logs every invocation (initial sign-in *and* subsequent SSR
  // calls) so we can see whether the callback runs at all, and what
  // shape the token / profile is in. Remove once /portal is reliably
  // populating session.user.
  console.log("[auth] jwtCallback", {
    hasAccount: !!account,
    accountProvider: account?.provider,
    accountType: account?.type,
    profileKeys: profile ? Object.keys(profile) : null,
    profile,
    user,
    tokenSub: token.sub,
    tokenName: token.name,
    tokenEmail: token.email,
    tokenGroupsCount: Array.isArray(token.groups) ? token.groups.length : null,
  });
  if (account) {
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
  // Temporary diagnostic — see jwtCallback comment above.
  console.log("[auth] sessionCallback", {
    tokenSub: token.sub,
    tokenName: token.name,
    tokenEmail: token.email,
    tokenGroupsCount: Array.isArray(token.groups) ? token.groups.length : null,
    hasAccessToken: typeof token.accessToken === "string",
  });
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
