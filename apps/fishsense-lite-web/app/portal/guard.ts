import { redirect } from "next/navigation";
import { auth } from "@/auth";
import { isPortalAuthorized } from "@/lib/authz";

/**
 * The gate every portal page under the index shares.
 *
 * Authenticated is not authorized — signing in only proves an account in the
 * Authentik realm. An unauthorized user is sent to `/portal`, which is the one
 * page that explains *why* they cannot get in and offers a sign-out; sending
 * them to the sign-in flow instead would loop forever for someone who is
 * already signed in and simply lacks the group.
 *
 * This is a rendering decision, not the security boundary. Server actions are
 * public endpoints and re-check the session themselves — see
 * `calibration/actions.ts`.
 */
export async function requirePortalUser(path: string) {
  const session = await auth();
  if (!session?.user) {
    redirect(`/api/auth/signin?callbackUrl=${encodeURIComponent(path)}`);
  }
  if (!isPortalAuthorized(session)) {
    redirect("/portal");
  }
  return session.user;
}
