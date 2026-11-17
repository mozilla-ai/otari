import type { McpServerCheckFailure, WorkspaceMcpServerHealth } from "@/client"
import { isLiteralToolName } from "@/features/tools/mcpAccess"

// What the gateway last learned by talking to a server, in words: from a
// manager's check, or from a real request that reached it. The codes are
// `CheckFailure` in `services/mcp_stateless.py`; each names a cause the
// operator can act on, so each gets a sentence
// saying what to do about it.

const FAILURES: Record<
  McpServerCheckFailure,
  { detail: (status: number | null | undefined) => string }
> = {
  unsafe_url: {
    detail: () =>
      "The URL now resolves to an address the gateway may not connect to.",
  },
  credentials_unavailable: {
    detail: () =>
      "The stored token cannot be decrypted, which usually means OTARI_SECRET_KEY changed. Set the token again.",
  },
  unreachable: {
    detail: () =>
      "The gateway could not connect to the host. Check the URL and that the server is running.",
  },
  tls_failed: {
    detail: () => "The TLS handshake failed. Check the server's certificate.",
  },
  timeout: {
    detail: () => "The server did not answer in time.",
  },
  redirected: {
    detail: (status) =>
      `The server answered with a redirect (HTTP ${status ?? "3xx"}), which the gateway does not follow. Use the URL it redirects to.`,
  },
  unauthorized: {
    detail: () =>
      "The server rejected the request as unauthenticated (HTTP 401). Check the authorization token.",
  },
  forbidden: {
    detail: () =>
      "The server refused access (HTTP 403). Check what the token is allowed to do.",
  },
  not_found: {
    detail: (status) =>
      `There is no MCP endpoint at this URL (HTTP ${status ?? 404}). Check the path.`,
  },
  server_error: {
    detail: (status) =>
      `The server failed with an error of its own (HTTP ${status ?? "5xx"}).`,
  },
  http_error: {
    detail: (status) =>
      `The server refused the request (HTTP ${status ?? "4xx"}).`,
  },
  not_mcp: {
    detail: () => "The URL answered, but not as a streamable HTTP MCP server.",
  },
  limits_exceeded: {
    detail: () => "The server's answer exceeded the gateway's limits.",
  },
  connection_failed: {
    detail: () => "The connection failed.",
  },
  sign_in_unsupported: {
    detail: () =>
      "Its authorization server does not offer S256 PKCE, which signing in requires, so nobody can connect an account.",
  },
  sign_in_unavailable: {
    detail: () =>
      "Its sign-in setup could not be read, so nobody can connect an account yet. Check the URL, or that the server supports OAuth sign-in.",
  },
}

/**
 * How many tools the allow-list admits, and how many the server offers,
 * read as the admitted count where the check recorded no names.
 */
function toolCounts(health: WorkspaceMcpServerHealth) {
  const admitted = health.tool_count ?? 0
  return { admitted, offered: health.tools?.length || admitted }
}

function toolNoun(count: number) {
  return count === 1 ? "tool" : "tools"
}

/** Whether the gateway holds a listing of the server's tools, from a passing check or a request that read them. */
export function hasToolListing(
  health: WorkspaceMcpServerHealth | null,
): health is WorkspaceMcpServerHealth {
  return (
    health?.status === "healthy" &&
    health.tool_count !== null &&
    health.tool_count !== undefined
  )
}

/** One sentence saying what the check found and, on a failure, what to fix. */
export function healthDetail(health: WorkspaceMcpServerHealth): string {
  if (health.error) {
    return FAILURES[health.error].detail(health.http_status)
  }
  const missing = health.missing_allowed_tools ?? []
  if (missing.length > 0) {
    return `The server does not offer ${missing.join(", ")}, so the allow-list exposes ${health.tool_count ?? 0} of its names.`
  }
  // A request reached the server without listing its tools, and no listing is on record.
  if (!hasToolListing(health)) return "Connected."
  const { admitted, offered } = toolCounts(health)
  return admitted === offered
    ? `Connected. The server offers ${offered} ${toolNoun(offered)}.`
    : `Connected. The allow-list admits ${admitted} of the ${offered} tools the server offers.`
}

/** Whether a check found something the operator should look at. */
export function healthNeedsAttention(health: WorkspaceMcpServerHealth) {
  return (
    health.status === "failing" ||
    (health.missing_allowed_tools ?? []).length > 0
  )
}

/**
 * How long a server may go without answering before its dot turns amber. Every
 * request that reaches a server refreshes this, so only one nobody uses gets
 * here, and the manual Check is for those.
 */
export const STALE_AFTER_MS = 7 * 24 * 60 * 60 * 1000

export type HealthState = "healthy" | "attention" | "failing" | "unchecked"

/**
 * The four states the server list's dot shows: never reached, failing,
 * answering but worth a look (no successful contact in a week, or allow-listed
 * tools the server no longer offers), and healthy.
 */
export function healthState(
  health: WorkspaceMcpServerHealth | null,
  now: number = Date.now(),
): HealthState {
  if (health === null) return "unchecked"
  if (health.status === "failing") return "failing"
  const checkedAt = Date.parse(health.checked_at)
  if (
    healthNeedsAttention(health) ||
    (!Number.isNaN(checkedAt) && now - checkedAt > STALE_AFTER_MS)
  ) {
    return "attention"
  }
  return "healthy"
}

export const HEALTH_STATE: Record<HealthState, { label: string; dot: string }> =
  {
    healthy: { label: "Healthy", dot: "bg-success" },
    attention: { label: "Needs attention", dot: "bg-warning" },
    failing: { label: "Failing", dot: "bg-danger" },
    unchecked: { label: "Not checked", dot: "bg-text-subtle" },
  }

/** What the dot means for one server, in a sentence. */
export function healthStateDetail(
  health: WorkspaceMcpServerHealth | null,
  state: HealthState,
): string {
  if (health === null) {
    return "Not checked yet. It is checked on its own the first time a request uses it."
  }
  if (state === "attention" && !healthNeedsAttention(health)) {
    return "No successful contact in the last 7 days. Requests that use this server refresh this on their own; check it by hand if none do."
  }
  return healthDetail(health)
}

/**
 * The tools a server last listed, for an editor to pick from: literal names
 * only, with their descriptions. `isKnown` is false until a check or a request
 * has listed them, when an editor falls back to typed names. With
 * `withinAllowList`, only the tools the server's allow-list exposes.
 */
export function checkedTools(
  server: {
    health: WorkspaceMcpServerHealth | null
    allowed_tools: readonly string[] | null
  },
  { withinAllowList = false }: { withinAllowList?: boolean } = {},
): {
  isKnown: boolean
  tools: string[]
  descriptions: Readonly<Record<string, string>>
} {
  const health = server.health
  const isKnown = hasToolListing(health)
  const offered = (health?.tools ?? []).filter(isLiteralToolName)
  const ceiling = withinAllowList ? server.allowed_tools : null
  return {
    isKnown,
    tools: isKnown
      ? ceiling === null
        ? offered
        : offered.filter((name) => ceiling.includes(name))
      : [],
    descriptions: health?.tool_descriptions ?? {},
  }
}
