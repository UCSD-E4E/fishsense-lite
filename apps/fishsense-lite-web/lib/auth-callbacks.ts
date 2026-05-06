import type { Account, Profile, Session } from "next-auth";
import type { JWT } from "next-auth/jwt";

interface AuthentikProfileLike extends Profile {
  groups?: string[];
}

interface JwtCallbackArgs {
  token: JWT;
  account?: Account | null;
  profile?: AuthentikProfileLike;
}

export async function jwtCallback({ token, account, profile }: JwtCallbackArgs): Promise<JWT> {
  if (account) {
    if (typeof account.access_token === "string") {
      token.accessToken = account.access_token;
    }
    token.groups = Array.isArray(profile?.groups) ? profile.groups : [];
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
  session.user.groups = Array.isArray(token.groups) ? token.groups : [];
  return session;
}
