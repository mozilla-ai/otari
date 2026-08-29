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

const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]"])

/**
 * Whether this page is a local dashboard, the case that may announce a missing
 * Mixpanel key.
 *
 * `make dev` serves a production Vite build on localhost, so `import.meta.env.DEV`
 * is false there and cannot be the only signal. A deployed OSS host is neither
 * DEV nor a loopback name, and must stay silent.
 */
export function isLocalDashboard(
  hostname: string = typeof window === "undefined"
    ? ""
    : window.location.hostname,
  isDev: boolean = import.meta.env.DEV,
): boolean {
  return isDev || LOCAL_HOSTS.has(hostname)
}
