import { NextResponse } from "next/server";
import { auth } from "@/auth";
import { isPortalAuthorized } from "@/lib/authz";
import { fetchTaskImage, getTask } from "@/lib/label-studio-tasks";

/**
 * Streams a task's frame through this server.
 *
 * Two reasons this is a proxy rather than a redirect. Label Studio's
 * `resolve` endpoint is on its API server and is **authenticated**, and the
 * bearer it wants expires in about five minutes — shorter than a labeling
 * session, so a URL handed to the browser would go stale mid-queue. Fetching
 * server-side means the client never holds a Label Studio credential at all.
 *
 * It takes only a task id, never a `fileuri`. Accepting a caller-supplied URI
 * would make this an open proxy that fetches anything the server can reach;
 * resolving the URI from the task itself keeps the reachable set to frames
 * that already belong to a project.
 */
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ taskId: string }> },
) {
  const session = await auth();
  if (!session?.user || !isPortalAuthorized(session)) {
    return new NextResponse("Forbidden", { status: 403 });
  }

  const { taskId: raw } = await params;
  const taskId = Number(raw);
  if (!Number.isInteger(taskId) || taskId <= 0) {
    return new NextResponse("Bad task id", { status: 400 });
  }

  const task = await getTask(taskId);
  const uri = typeof task?.data?.image === "string" ? task.data.image : null;
  if (!uri) {
    return new NextResponse("Task has no image", { status: 404 });
  }

  const upstream = await fetchTaskImage(taskId, uri);
  if (!upstream.ok || !upstream.body) {
    return new NextResponse("Upstream image fetch failed", { status: 502 });
  }

  return new NextResponse(upstream.body, {
    status: 200,
    headers: {
      "content-type": upstream.headers.get("content-type") ?? "image/jpeg",
      // Frames are immutable once written, and a labeler revisits the same one
      // while zooming. Private because the bytes are behind portal auth.
      "cache-control": "private, max-age=3600",
    },
  });
}
