import { env } from "./env";

/** The dive fields the portal needs. The API returns more (path, camera_id,
 * flip_dive_slate); we only type what we render or act on. */
export type Dive = {
  id: number;
  name: string | null;
  dive_datetime: string | null;
  priority: string | null;
  dive_slate_id: number | null;
  calibration_dive_id: number | null;
};

function basicAuthHeader(): string {
  const token = Buffer.from(
    `${env.fishsenseApiUsername}:${env.fishsenseApiPassword}`,
  ).toString("base64");
  return `Basic ${token}`;
}

/** Validate a dive id before it goes into a request URL.
 *
 * These ids originate client-side (a `<select>` value passed through a server
 * action), so TypeScript's `number` type is no runtime guarantee. Coercing
 * through `Number()` and constraining to a non-negative integer stops anything
 * untrusted from injecting extra path segments or steering the request
 * elsewhere (js/request-forgery). The returned value is a plain number — safe
 * to interpolate as a single path segment. */
function safeId(value: number, label: string): number {
  const id = Number(value);
  if (!Number.isInteger(id) || id < 0) {
    throw new Error(`Invalid ${label}: ${value}`);
  }
  return id;
}

/** Every dive, for the calibration-linking table. */
export async function getDives(revalidate = 0): Promise<Dive[]> {
  const url = `${env.fishsenseApiUrl}/api/v1/dives/`;
  const response = await fetch(url, {
    headers: { Authorization: basicAuthHeader() },
    next: { revalidate },
  });

  if (!response.ok) {
    console.error("[dives] list fetch failed", {
      url,
      status: response.status,
      statusText: response.statusText,
    });
    throw new Error(
      `fishsense-api dives list failed: ${response.status} ${response.statusText}`,
    );
  }

  return (await response.json()) as Dive[];
}

/** Link `diveId` to borrow `sourceId`'s laser calibration. */
export async function setCalibrationSource(
  diveId: number,
  sourceId: number,
): Promise<void> {
  const url = `${env.fishsenseApiUrl}/api/v1/dives/${safeId(diveId, "diveId")}/calibration-source/${safeId(sourceId, "sourceId")}`;
  const response = await fetch(url, {
    method: "PUT",
    headers: { Authorization: basicAuthHeader() },
    cache: "no-store",
  });

  if (!response.ok) {
    console.error("[dives] set calibration source failed", {
      url,
      status: response.status,
      statusText: response.statusText,
    });
    throw new Error(
      `set calibration source failed: ${response.status} ${response.statusText}`,
    );
  }
}

/** Remove any borrowed-calibration link from `diveId` (idempotent). */
export async function clearCalibrationSource(diveId: number): Promise<void> {
  const url = `${env.fishsenseApiUrl}/api/v1/dives/${safeId(diveId, "diveId")}/calibration-source/`;
  const response = await fetch(url, {
    method: "DELETE",
    headers: { Authorization: basicAuthHeader() },
    cache: "no-store",
  });

  if (!response.ok) {
    console.error("[dives] clear calibration source failed", {
      url,
      status: response.status,
      statusText: response.statusText,
    });
    throw new Error(
      `clear calibration source failed: ${response.status} ${response.statusText}`,
    );
  }
}
