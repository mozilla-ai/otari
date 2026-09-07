/**
 * The Mixpanel project token the dashboard was built with, if any.
 *
 * Vite only exposes `VITE_*` to the client, and it inlines the value at build
 * time (`make dashboard`, `pnpm run dev`, the Docker web stage). An empty or
 * whitespace-only value is treated as absent: that is the OSS default, and it
 * is what keeps the SDK from loading.
 */
export function readMixpanelToken(
  raw: unknown = import.meta.env.VITE_MIXPANEL_TOKEN,
): string | undefined {
  if (typeof raw !== "string") {
    return undefined
  }
  const token = raw.trim()
  return token === "" ? undefined : token
}

/**
 * Whether this page is a dev-server dashboard, the one case that may announce a
 * missing Mixpanel key.
 *
 * Deliberately the Vite DEV flag alone, and not a loopback hostname: a
 * self-hosted gateway is a production build on `http://localhost:8000/` (the
 * address `README.md` and `docs/dashboard.md` both tell operators to open), so
 * a hostname test cannot tell an OSS operator from a mozilla.ai developer and
 * would tell every operator that Mixpanel failed to start. `make dev` serves a
 * production build and so loses the reminder; `pnpm run dev` still has it.
 */
export function isLocalDashboard(
  isDev: boolean = import.meta.env.DEV,
): boolean {
  return isDev
}
