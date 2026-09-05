"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ProjectOutcome, TriageItem } from "@/lib/triage-queue";
import { acceptAction, undoAcceptAction } from "./actions";

/**
 * Deliberately very high. A 4000px-wide frame fitted to a phone screen puts one
 * image pixel at roughly 0.1 CSS px, so reaching a pixel you can actually look
 * at needs a scale near 100. Verified on device: at 120x with dpr 2.81 the
 * frame stays hard-edged, one image pixel covering ~160 device pixels.
 */
const MAX_SCALE = 120;
const MIN_SCALE = 0.02;

type Props = {
  items: TriageItem[];
  kindLabel: string;
  /** What each walked project contributed — see the empty state. */
  scanned: number;
  projects: ProjectOutcome[];
  notWalked: number;
};

export function TriageClient({ items, kindLabel, scanned, projects, notWalked }: Props) {
  const [index, setIndex] = useState(0);
  const [accepted, setAccepted] = useState(0);
  const [skipped, setSkipped] = useState(0);
  const [message, setMessage] = useState<string | null>(null);
  const [undo, setUndo] = useState<{ taskId: number; annotationId: number } | null>(null);

  const item = items[index] ?? null;
  const startedAt = useRef(0);

  useEffect(() => {
    startedAt.current = Date.now();
  }, [index]);

  const advance = useCallback(() => {
    setIndex((i) => i + 1);
    setMessage(null);
  }, []);

  async function onAccept() {
    if (!item) return;
    const leadTime = Date.now() - startedAt.current;
    const current = item;
    setAccepted((n) => n + 1);
    advance();

    const res = await acceptAction(current.taskId, current.result, leadTime);
    if (res.ok) {
      // Undo stays available after the queue has moved on — a mis-tap is
      // usually noticed a beat later, once the next frame is up.
      setUndo({ taskId: current.taskId, annotationId: res.annotationId });
    } else {
      // Must roll the count back: without it a failed accept still read as
      // accepted, which is the number a labeler trusts.
      setAccepted((n) => n - 1);
      setMessage(`Accept failed for task ${current.taskId}: ${res.error}`);
    }
  }

  function onSkip() {
    if (!item) return;
    // Client-side only. A Label Studio "skip" writes a cancelled annotation
    // with an empty result, which flips `completed` with no coordinates and
    // makes the populate step exclude the image forever.
    setSkipped((n) => n + 1);
    advance();
  }

  async function onUndo() {
    if (!undo) return;
    const res = await undoAcceptAction(undo.annotationId);
    setUndo(null);
    if (res.ok) {
      setAccepted((n) => Math.max(0, n - 1));
      setMessage(`Took back task ${undo.taskId}`);
    } else {
      setMessage(res.error);
    }
  }

  if (!item) {
    return (
      <div className="mx-auto flex w-full max-w-xl flex-1 flex-col justify-center gap-3 p-8">
        <p className="text-sm text-slate-300">
          Nothing left to triage in the {kindLabel} queue.
        </p>

        {/* An empty queue and a broken one looked identical, which cost real
            time to tell apart more than once. `loadQueue` already knew all of
            this and the page was throwing it away. */}
        <p className="text-xs text-slate-500">
          Walked {scanned} {scanned === 1 ? "project" : "projects"}.{" "}
          {scanned === 0
            ? "No projects were offered — either Label Studio is switched off, or the auto-accept gate has not finished sweeping any dive yet."
            : "Every task in them was already handled."}
        </p>

        {projects.length > 0 && (
          <details className="text-xs text-slate-500" open>
            <summary className="cursor-pointer">Per project</summary>
            <ul className="mt-2 space-y-1 font-mono text-[11px] text-slate-400">
              {projects.map((p) => (
                <li key={p.projectId}>
                  <span className="text-slate-300">{p.projectId}</span>
                  {p.title ? ` ${p.title}` : ""}
                  {" — "}
                  {p.error
                    ? p.error
                    : p.tasks === 0
                      ? "0 tasks on page 1"
                      : `${p.tasks} tasks, ${p.taken} taken` +
                        (Object.keys(p.reasons).length > 0
                          ? ` (${Object.entries(p.reasons)
                              .map(([reason, n]) => `${n} ${reason}`)
                              .join(", ")})`
                          : "")}
                </li>
              ))}
            </ul>
            {notWalked > 0 && (
              <p className="mt-2 text-[11px] text-slate-500">
                {notWalked} more offered but not reached this load.
              </p>
            )}
          </details>
        )}

        <p className="text-xs text-slate-600">
          {accepted} accepted · {skipped} skipped this session
        </p>
      </div>
    );
  }

  return (
    <>
      <Viewer key={item.taskId} item={item} />

      <footer className="border-t border-slate-800 px-4 py-3">
        <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
          <span className="truncate" title={item.diveName}>
            {item.diveName} · task {item.taskId}
          </span>
          <span className="tabular-nums">
            {accepted} accepted · {skipped} skipped · {items.length - index - 1} left
          </span>
        </div>

        {item.partial && (
          <p className="mb-2 rounded border border-amber-700/60 bg-amber-900/25 px-2 py-1 text-xs text-amber-300">
            Partial detection — the model found {item.keypoints.length} of the expected points.
          </p>
        )}

        {message && (
          <p className="mb-2 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-300">
            {message}
          </p>
        )}

        <div className="flex gap-2">
          <button
            type="button"
            onClick={onSkip}
            className="flex-1 rounded border border-slate-700 bg-slate-900 py-3 text-sm font-medium hover:bg-slate-800"
          >
            Skip
          </button>
          <button
            type="button"
            onClick={onAccept}
            className="flex-[2] rounded border border-emerald-600 bg-emerald-600/90 py-3 text-sm font-semibold text-emerald-950 hover:bg-emerald-500"
          >
            Accept
          </button>
          {undo && (
            <button
              type="button"
              onClick={onUndo}
              className="rounded border border-slate-700 bg-slate-900 px-3 text-xs hover:bg-slate-800"
            >
              Undo
            </button>
          )}
        </div>
      </footer>
    </>
  );
}

/**
 * Zoomable frame with constant-size prediction markers.
 *
 * The markers are carried by the image — they name a specific pixel, so they
 * must travel with it — while staying a fixed size on screen. That is what
 * makes deep zoom useful: at high magnification the ring ends up smaller than
 * a single image pixel, so the pixel being claimed is unambiguous.
 */
function Viewer({ item }: { item: TriageItem }) {
  const stageRef = useRef<HTMLDivElement>(null);
  const layerRef = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  const view = useRef({ k: 1, tx: 0, ty: 0 });

  const first = item.keypoints[0];
  const W = first?.originalWidth ?? 4000;
  const H = first?.originalHeight ?? 3000;

  /**
   * One imperative write per frame, and no React state.
   *
   * The markers live INSIDE the transformed layer, so they are carried by the
   * image for free. `--inv` counter-scales them, which is what keeps the ring
   * 11 CSS px across at every magnification: the browser composites it, so
   * panning never re-renders the tree or recomputes a screen position.
   */
  const paint = useCallback(() => {
    const layer = layerRef.current;
    if (!layer) return;
    const { k, tx, ty } = view.current;
    layer.style.transform = `translate(${tx}px,${ty}px) scale(${k})`;
    layer.style.setProperty("--inv", String(1 / k));
  }, []);

  const fit = useCallback(() => {
    const stage = stageRef.current;
    if (!stage) return;
    // Fractional box on purpose: `clientHeight` is integer-rounded, and that
    // rounding is visible once the frame is magnified a hundredfold.
    const r = stage.getBoundingClientRect();
    const k = Math.min(r.width / W, r.height / H);
    view.current = { k, tx: (r.width - W * k) / 2, ty: (r.height - H * k) / 2 };
    paint();
  }, [W, H, paint]);

  useEffect(() => {
    fit();
    const onResize = () => fit();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [fit]);

  // --- gestures -------------------------------------------------------------

  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const pinch = useRef<{ dist: number; cx: number; cy: number } | null>(null);
  const lastTap = useRef(0);

  function pinchState() {
    const p = [...pointers.current.values()];
    return {
      dist: Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y) || 1,
      cx: (p[0].x + p[1].x) / 2,
      cy: (p[0].y + p[1].y) / 2,
    };
  }

  function zoomAbout(nextK: number, cx: number, cy: number) {
    const { k, tx, ty } = view.current;
    const ix = (cx - tx) / k;
    const iy = (cy - ty) / k;
    const clamped = Math.max(MIN_SCALE, Math.min(MAX_SCALE, nextK));
    view.current = { k: clamped, tx: cx - ix * clamped, ty: cy - iy * clamped };
    paint();
  }

  return (
    <div
      ref={stageRef}
      className="relative flex-1 touch-none select-none overflow-hidden bg-black"
      onPointerDown={(e) => {
        (e.target as Element).setPointerCapture?.(e.pointerId);
        pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
        if (pointers.current.size === 2) pinch.current = pinchState();

        const now = Date.now();
        if (pointers.current.size === 1 && now - lastTap.current < 320) {
          const r = stageRef.current!.getBoundingClientRect();
          zoomAbout(view.current.k > 60 ? 1 : MAX_SCALE, e.clientX - r.left, e.clientY - r.top);
        }
        lastTap.current = now;
      }}
      onPointerMove={(e) => {
        if (!pointers.current.has(e.pointerId)) return;
        const prev = pointers.current.get(e.pointerId)!;
        pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });

        if (pointers.current.size === 1) {
          view.current.tx += e.clientX - prev.x;
          view.current.ty += e.clientY - prev.y;
          paint();
        } else if (pointers.current.size === 2 && pinch.current) {
          const now = pinchState();
          const r = stageRef.current!.getBoundingClientRect();
          const { k, tx, ty } = view.current;
          const ix = (pinch.current.cx - r.left - tx) / k;
          const iy = (pinch.current.cy - r.top - ty) / k;
          const nk = Math.max(
            MIN_SCALE,
            Math.min(MAX_SCALE, k * (now.dist / pinch.current.dist)),
          );
          view.current = {
            k: nk,
            tx: now.cx - r.left - ix * nk,
            ty: now.cy - r.top - iy * nk,
          };
          pinch.current = now;
          paint();
        }
      }}
      onPointerUp={(e) => {
        pointers.current.delete(e.pointerId);
        pinch.current = pointers.current.size === 2 ? pinchState() : null;
      }}
      onPointerCancel={(e) => {
        pointers.current.delete(e.pointerId);
        pinch.current = null;
      }}
    >
      <div ref={layerRef} className="absolute left-0 top-0 origin-top-left">
        {/* Deliberately a plain <img>. `next/image` resamples and re-encodes,
            which is exactly what must not happen to a frame whose purpose is
            judging an individual pixel — and the bytes come from an
            authenticated proxy the optimizer cannot reach anyway. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`/api/triage/image/${item.taskId}`}
          alt=""
          draggable={false}
          width={W}
          height={H}
          // `max-w-none` is load-bearing: Tailwind's preflight sets
          // `img { max-width: 100% }`, which clamps a 4000px frame to the stage
          // width so `scale(k)` multiplies a base ten times too small.
          // Smoothing is off because it invents detail that is not in the
          // frame, the opposite of what a verification view should do.
          className="pointer-events-none block max-w-none [image-rendering:pixelated]"
          onLoad={() => setReady(true)}
        />

        {/* Held back until a frame has actually decoded: drawn over a blank
            view, a marker claims a point on an image nobody can see. */}
        {ready &&
          item.keypoints.map((kp, i) => (
            <svg
              key={i}
              viewBox="-20 -20 40 40"
              aria-hidden
              className="pointer-events-none absolute h-10 w-10 overflow-visible"
              style={{
                left: `${(kp.xPercent / 100) * W}px`,
                top: `${(kp.yPercent / 100) * H}px`,
                margin: "-20px 0 0 -20px",
                transform: "scale(var(--inv, 1))",
              }}
            >
              <circle r="11" fill="none" stroke="#000" strokeOpacity="0.55" strokeWidth="3.5" />
              <circle r="11" fill="none" stroke={colourFor(kp.label)} strokeWidth="1.5" />
              <path
                d="M-9 0H-4M4 0H9M0 -9V-4M0 4V9"
                stroke={colourFor(kp.label)}
                strokeWidth="1.2"
              />
              <circle r="1" fill={colourFor(kp.label)} />
            </svg>
          ))}
      </div>

      {/* Outside the transformed layer, so retained zoom cannot magnify it. */}
      {!ready && (
        <div className="absolute inset-0 grid place-items-center text-xs text-slate-500">
          Loading frame…
        </div>
      )}
    </div>
  );
}

function colourFor(label: string): string {
  const l = label.toLowerCase();
  if (l.includes("green")) return "#41e06a";
  if (l.includes("red")) return "#ff5252";
  if (l.includes("snout")) return "#ffa39e";
  if (l.includes("fork")) return "#26a269";
  return "#ffd400";
}
