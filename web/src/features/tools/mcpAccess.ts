import type {
  WorkspaceMcpServerAccessCheck,
  WorkspaceMcpServerKeyAccess,
} from "@/client"

// A stored MCP server's access list, in words. The grammar is the gateway's
// (`services/tools/_mcp_admission.py`): a rule is a list of tool patterns,
// where `*` is every tool, `prefix*` every tool starting so, `@read` the tools
// a manager labeled read on that server, any other entry one tool by name, and
// `[]` denies.

export const ALL_TOOLS = "*"
export const READ_TOOLS = "@read"

/**
 * Whether a server's tool name means only itself once picked, the gateway's
 * own rule (`reads_as_pattern`): not empty, no wildcard, comma or leading `@`,
 * and no whitespace or control character anywhere, since this side and the
 * gateway disagree on what trimming removes. Anything else is never offered as
 * a choice; a manager types a pattern on purpose instead.
 */
export function isLiteralToolName(name: string): boolean {
  return (
    name !== "" &&
    !name.includes("*") &&
    !name.includes(",") &&
    !name.startsWith("@") &&
    !/[\s\p{C}\p{Z}]/u.test(name)
  )
}

/** The four answers a rule can give, as the page offers them. */
export type AccessMode = "all" | "read" | "none" | "some"

export function accessMode(tools: readonly string[]): AccessMode {
  if (tools.includes(ALL_TOOLS)) return "all"
  if (tools.length === 1 && tools[0] === READ_TOOLS) return "read"
  return tools.length === 0 ? "none" : "some"
}

/** One pattern as the page names it: `@read` reads "read tools". */
function patternLabel(pattern: string): string {
  return pattern === READ_TOOLS ? "read tools" : pattern
}

/** The patterns a mode stands for; `picked` is what "some" allows. */
export function toolsFor(
  mode: AccessMode,
  picked: readonly string[],
): string[] {
  if (mode === "all") return [ALL_TOOLS]
  if (mode === "read") return [READ_TOOLS]
  if (mode === "none") return []
  return [...picked]
}

/** The draft an `AccessChooser` holds for a stored rule. */
export function accessDraft(tools: readonly string[]): {
  mode: AccessMode
  picked: string[]
} {
  const mode = accessMode(tools)
  return { mode, picked: mode === "some" ? [...tools] : [] }
}

/** Whether a draft can be saved: "some" needs at least one tool. */
export function isDraftComplete(
  mode: AccessMode,
  picked: readonly string[],
): boolean {
  return mode !== "some" || picked.length > 0
}

/** A rule's action, as the ACL table reads it: `[]` denies, anything else allows. */
export function ruleAction(tools: readonly string[]): "ALLOW" | "DENY" {
  return tools.length === 0 ? "DENY" : "ALLOW"
}

/** The tools a rule matches, as the ACL table reads them. */
export function ruleTools(tools: readonly string[]): string {
  const mode = accessMode(tools)
  if (mode === "read") return "Read tools"
  return mode === "some" ? tools.map(patternLabel).join(", ") : "Any"
}

/** A rule in one phrase: "allow any tool", "deny", "allow get_issue". */
export function ruleSummary(tools: readonly string[]): string {
  const mode = accessMode(tools)
  if (mode === "all") return "allow any tool"
  if (mode === "read") return "allow read tools"
  if (mode === "none") return "deny"
  return `allow ${tools.map(patternLabel).join(", ")}`
}

/** The list page's summary of a server's default policy, uppercase like its cells. */
export function defaultPolicySummary(tools: readonly string[]): string {
  const mode = accessMode(tools)
  if (mode === "all") return "DEFAULT: ALLOW"
  if (mode === "read") return "DEFAULT: READ TOOLS"
  if (mode === "none") return "DEFAULT: DENY"
  return `DEFAULT: ALLOW ${tools.length} ${tools.length === 1 ? "TOOL" : "TOOLS"}`
}

/** Whether one pattern admits a tool name. */
export function patternMatches(pattern: string, tool: string): boolean {
  if (pattern === READ_TOOLS) return false
  if (pattern === ALL_TOOLS) return true
  if (pattern.endsWith("*")) return tool.startsWith(pattern.slice(0, -1))
  return pattern === tool
}

/**
 * What a key reaches on one server, for the key editor: the deciding rule cut
 * down by the server's own ceiling, since a tool the server does not expose is
 * never reachable whatever the rule says.
 */
export function keyAccessLabel(access: WorkspaceMcpServerKeyAccess): string {
  const mode = accessMode(access.tools)
  if (mode === "none") return "No access"
  const ceiling = access.allowed_tools
  if (mode === "all") {
    return ceiling === null ? "All tools" : ceiling.join(", ") || "No access"
  }
  // `@read` stands for the server's labeled read tools, which reach exactly as named ones do.
  const patterns = access.tools.flatMap((pattern) =>
    pattern === READ_TOOLS ? (access.read_tools ?? []) : [pattern],
  )
  const reachable =
    ceiling === null
      ? patterns
      : patterns.flatMap((pattern) =>
          pattern.endsWith("*")
            ? ceiling.filter((tool) => patternMatches(pattern, tool))
            : ceiling.includes(pattern)
              ? [pattern]
              : [],
        )
  const names = [...new Set(reachable)].join(", ")
  if (!names) return "No access"
  return mode === "read" ? `Read tools: ${names}` : names
}

/** Which rule decided a key's access, as the key editor says it. */
export function decidedByLabel(
  decidedBy: WorkspaceMcpServerKeyAccess["decided_by"],
): string {
  if (decidedBy === "key") return "the rule for this key"
  if (decidedBy === "user") return "the rule for its owner"
  return "the default policy"
}

/**
 * A trace's verdict and the rule that matched, as one line:
 * "ALLOW, matched the rule for Bo's laptop". `ruleOwner` names the grant that
 * matched, which only the page can label.
 */
export function verdictLabel(
  check: WorkspaceMcpServerAccessCheck,
  { ruleOwner, serverEnabled }: { ruleOwner: string; serverEnabled: boolean },
): string {
  const verdict = check.allowed ? "ALLOW" : "DENY"
  if (check.decided_by === "server") {
    return serverEnabled
      ? `${verdict}, the server does not expose this tool`
      : `${verdict}, the server is disabled`
  }
  if (check.decided_by === "everyone") {
    return `${verdict}, matched the default policy (${ruleSummary(check.tools)})`
  }
  return `${verdict}, matched the rule for ${ruleOwner}`
}
