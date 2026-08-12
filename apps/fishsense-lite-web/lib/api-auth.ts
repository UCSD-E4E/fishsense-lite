import { env } from "./env";

/** `Authorization` header for fishsense-api's HTTP Basic auth.
 *
 * SSR reaches the API on the interior docker network (`http://fishsense-api:8000`),
 * bypassing the traefik forwardAuth gate — see the FISHSENSE_API_URL comment in
 * `deploy/incus/compose.yml`. Built per call rather than cached at module load
 * so `env`'s per-request lazy read still applies.
 */
export function fishsenseApiAuthHeader(): string {
  const token = Buffer.from(
    `${env.fishsenseApiUsername}:${env.fishsenseApiPassword}`,
  ).toString("base64");
  return `Basic ${token}`;
}
