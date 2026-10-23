import type { McpServerCheckFailure, WorkspaceMcpServerHealth } from "@/client"

// What the gateway's connection check found, in words. The codes are
// `CheckFailure` in `services/mcp_stateless.py`; each names a cause the
// operator can act on, so each gets a short label for the table and a sentence
// saying what to do about it.

const FAILURES: Record<
  McpServerCheckFailure,
  { label: string; detail: (status: number | null | undefined) => string }
> = {
  unsafe_url: {
    label: "UNSAFE URL",
    detail: () =>
      "The URL now resolves to an address the gateway may not connect to.",
  },
  credentials_unavailable: {
    label: "TOKEN UNREADABLE",
    detail: () =>
      "The stored token cannot be decrypted, which usually means OTARI_SECRET_KEY changed. Set the token again.",
  },
  unreachable: {
    label: "UNREACHABLE",
    detail: () =>
      "The gateway could not connect to the host. Check the URL and that the server is running.",
  },
  tls_failed: {
    label: "TLS FAILED",
    detail: () => "The TLS handshake failed. Check the server's certificate.",
  },
  timeout: {
    label: "TIMED OUT",
    detail: () => "The server did not answer in time.",
  },
  redirected: {
    label: "REDIRECTED",
    detail: (status) =>
      `The server answered with a redirect (HTTP ${status ?? "3xx"}), which the gateway does not follow. Use the URL it redirects to.`,
  },
  unauthorized: {
    label: "TOKEN REJECTED",
    detail: () =>
      "The server rejected the request as unauthenticated (HTTP 401). Check the authorization token.",
  },
  forbidden: {
    label: "FORBIDDEN",
    detail: () =>
      "The server refused access (HTTP 403). Check what the token is allowed to do.",
  },
  not_found: {
    label: "NOT FOUND",
    detail: (status) =>
      `There is no MCP endpoint at this URL (HTTP ${status ?? 404}). Check the path.`,
  },
  server_error: {
    label: "SERVER ERROR",
    detail: (status) =>
      `The server failed with an error of its own (HTTP ${status ?? "5xx"}).`,
  },
  http_error: {
    label: "HTTP ERROR",
    detail: (status) =>
      `The server refused the request (HTTP ${status ?? "4xx"}).`,
  },
  not_mcp: {
    label: "NOT MCP",
    detail: () => "The URL answered, but not as a streamable HTTP MCP server.",
  },
  limits_exceeded: {
    label: "TOO LARGE",
    detail: () => "The server's answer exceeded the gateway's limits.",
  },
  connection_failed: {
    label: "FAILED",
    detail: () => "The connection failed.",
  },
}

/**
 * The table's few words for a check, uppercase like the cells beside it. A
 * passing check says only that: the Tools column carries the count.
 */
export function healthLabel(health: WorkspaceMcpServerHealth): string {
  if (!health.error) return "CONNECTED"
  return health.error === "http_error" && health.http_status
    ? `HTTP ${health.http_status}`
    : FAILURES[health.error].label
}

/**
 * The Tools column's count from a passing check: every tool the server offers,
 * or how many of them the allow-list admits. Undefined when no check passed.
 */
export function toolCountLabel(
  health: WorkspaceMcpServerHealth | null,
  hasAllowList: boolean,
): string | undefined {
  if (health?.status !== "healthy") return undefined
  const admitted = health.tool_count ?? 0
  const offered = health.tools?.length || admitted
  const noun = offered === 1 ? "tool" : "tools"
  return hasAllowList
    ? `${admitted} of ${offered} ${noun}`
    : `${offered} ${noun}`
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
  const admitted = health.tool_count ?? 0
  const offered = health.tools?.length || admitted
  return admitted === offered
    ? `Connected. The server offers ${offered} ${offered === 1 ? "tool" : "tools"}.`
    : `Connected. The allow-list admits ${admitted} of the ${offered} tools the server offers.`
}

/** Whether a check found something the operator should look at. */
export function healthNeedsAttention(health: WorkspaceMcpServerHealth) {
  return (
    health.status === "failing" ||
    (health.missing_allowed_tools ?? []).length > 0
  )
}
