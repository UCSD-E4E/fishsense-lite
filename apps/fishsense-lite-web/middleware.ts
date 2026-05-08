import { auth } from "@/auth";

// Temporary diagnostic wrapper around the auth middleware. Logs every
// /portal request with whether `req.auth` carries a session, so we can
// distinguish "user never signed in" from "cookie present but
// decryption failed." Remove once /portal is reliably populating
// session.user.
export default auth((req) => {
  const session = req.auth;
  console.log("[middleware] /portal request", {
    pathname: req.nextUrl.pathname,
    hasAuth: !!session,
    user: session?.user
      ? {
          name: session.user.name,
          email: session.user.email,
          groups: session.user.groups,
        }
      : null,
    cookieNames: req.cookies.getAll().map((c) => c.name),
  });
});

export const config = {
  matcher: ["/portal/:path*"],
};
