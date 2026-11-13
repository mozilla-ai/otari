import type {
  McpCatalogServer,
  McpCredentialMode,
  WorkspaceMcpServer,
  WorkspaceMcpServerHealth,
} from "@/client"
import { healthDetail } from "@/features/tools/mcpServerHealth"

// What the Browse catalog flow needs of an entry. `name` is the catalog's own
// identifier, which is what makes the same server on Popular and in a search
// one pick.
export interface CatalogPick {
  name: string
  title: string
  description: string
  url: string
  requiresToken: boolean
  /** Where the gateway serves its logo, or "" when it found none. */
  logoUrl: string
  logoTone: McpCatalogServer["logo_tone"]
  /** Its maker's own server, on the maker's own domain, as far as the gateway can tell. */
  isOfficial: boolean
}

export function toCatalogPick(entry: McpCatalogServer): CatalogPick {
  return {
    name: entry.name,
    title: entry.title,
    description: entry.description ?? "",
    url: entry.url,
    requiresToken: entry.requires_token,
    logoUrl: entry.logo_url ?? "",
    logoTone: entry.logo_tone,
    isOfficial: entry.publisher_kind === "vendor",
  }
}

/** The host a pick's endpoint lives on, which is what "Official" is a claim about. */
export function hostOf(pick: CatalogPick): string {
  try {
    return new URL(pick.url).host
  } catch {
    return pick.url
  }
}

export interface ModeSuggestion {
  mode: McpCredentialMode
  /** Why, in a few words, or "" while there is nothing to say yet. */
  reason: string
}

/**
 * The connection mode a picked server starts on in the review step.
 *
 * A server whose registry entry asks for an Authorization header takes one
 * token, so Shared. Otherwise the answer to a check with no credential
 * decides: a server that refuses it wants each caller to sign in, so
 * Personal; one that answers needs nobody's account, so Shared. Until that
 * answer arrives, or when the check could not tell, Shared, which is what a
 * server added by URL starts on.
 */
export function suggestMode(
  pick: CatalogPick,
  probe: WorkspaceMcpServerHealth | undefined,
  isProbing: boolean,
): ModeSuggestion {
  if (pick.requiresToken) {
    return { mode: "shared", reason: "The registry says it takes a token." }
  }
  if (probe?.error === "unauthorized" || probe?.error === "forbidden") {
    return { mode: "per_user", reason: "It asks each person to sign in." }
  }
  if (probe?.status === "healthy") {
    return { mode: "shared", reason: "It answers without signing in." }
  }
  return {
    mode: "shared",
    reason: isProbing ? "Checking how it signs in." : "",
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
  if (server.credential_mode === "per_user" && !health) {
    return {
      tone: "success",
      label: "Added",
      detail: "Added. Each member connects their own account on its page.",
    }
  }
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
      label:
        server.credential_mode === "shared" ? "Needs setup" : "Needs sign-in",
      detail:
        server.credential_mode === "shared"
          ? "Added, but the server asks for an account. Set it up on its page: sign in for everyone, or paste a token."
          : "Added, but the server wants a credential before it answers.",
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
