import type {
  McpCatalogFeatured,
  McpCatalogServer,
  WorkspaceMcpServer,
} from "@/client"
import { healthDetail } from "@/features/tools/mcpServerHealth"

// What the Browse catalog flow needs of an entry, whether it came from the
// catalog or from the organization's featured shelf. `name` is the catalog's
// own identifier, which is what makes the same server on both lists one pick.
export interface CatalogPick {
  /** The catalog server's id, absent for an entry the catalog no longer lists. */
  serverId: string | undefined
  name: string
  title: string
  description: string
  url: string
  requiresToken: boolean
}

export function toCatalogPick(
  entry: McpCatalogServer | McpCatalogFeatured,
): CatalogPick {
  return {
    serverId:
      "catalog_server_id" in entry
        ? (entry.catalog_server_id ?? undefined)
        : entry.id,
    name: entry.name,
    title: entry.title,
    description: entry.description ?? "",
    url: entry.url,
    requiresToken: entry.requires_token,
  }
}

/** A URL as two spellings of one endpoint compare equal: host folded, no trailing slash. */
export function normalizeEndpoint(url: string): string {
  try {
    const parsed = new URL(url.trim())
    const path = parsed.pathname.replace(/\/+$/, "")
    return `${parsed.protocol}//${parsed.host}${path}${parsed.search}`
  } catch {
    return url.trim()
  }
}

// A stored server's name holds at most this many characters.
const MAX_NAME_LENGTH = 128

/**
 * A name for a new server that no server in `taken` has, starting from
 * `title`: "Sentry", then "Sentry 2". Names compare as the gateway stores
 * them, exactly, so only an identical one collides.
 */
export function findFreeName(
  title: string,
  taken: ReadonlySet<string>,
): string {
  const base = title.trim().slice(0, MAX_NAME_LENGTH) || "MCP server"
  if (!taken.has(base)) return base
  for (let suffix = 2; ; suffix += 1) {
    const tail = ` ${suffix}`
    const candidate = `${base.slice(0, MAX_NAME_LENGTH - tail.length)}${tail}`
    if (!taken.has(candidate)) return candidate
  }
}

export type AddTone = "success" | "warning" | "danger"

export interface AddOutcome {
  tone: AddTone
  /** A word or two for the row's status. */
  label: string
  /** One sentence saying what happened and what to do next, if anything. */
  detail: string
}

/** What adding one server came to, read from the connection check the create ran. */
export function describeAdded(server: WorkspaceMcpServer): AddOutcome {
  const health = server.health
  if (!health) {
    return {
      tone: "success",
      label: "Added",
      detail: "Added. Its connection has not been checked yet.",
    }
  }
  if (!health.error) {
    return { tone: "success", label: "Connected", detail: healthDetail(health) }
  }
  if (health.error === "unauthorized" || health.error === "forbidden") {
    return {
      tone: "warning",
      label: "Needs sign-in",
      detail:
        "Added, but the server wants a credential before it answers. Give it a token on its page.",
    }
  }
  return {
    tone: "warning",
    label: "Check failed",
    detail: `Added, but its connection check failed. ${healthDetail(health)}`,
  }
}

/** What a refused add came to. `message` is the gateway's own reason. */
export function describeRefusal(message: string): AddOutcome {
  return { tone: "danger", label: "Not added", detail: message }
}
