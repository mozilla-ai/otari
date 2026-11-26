import type { KeyScope } from "@/client"

/** Which part of the form a scope is offered under. */
export type ScopeGroupKey =
  | "models"
  | "files"
  | "batches"
  | "tools"
  | "mcp"
  | "memory"
  | "agents"
  | "usage"
  | "endUsers"

export type ScopeInfo = {
  label: string
  group: ScopeGroupKey
  /** Set where a leaked key holding the scope could do lasting harm. */
  risk?: string
  /** Set where leaving the scope off quietly turns something else off. */
  hint?: string
}

/**
 * Every scope, in the gateway's order (`types/key_scope.py`). A record rather
 * than a list so that a scope the gateway adds fails the typecheck until it is
 * described here and given a group: one missing would be dropped from a key on
 * its next save, or offered under no heading.
 */
export const SCOPE_INFO: Record<KeyScope, ScopeInfo> = {
  "inference:run": { label: "Call models", group: "models" },
  "files:read": { label: "List and download files", group: "files" },
  "files:write": {
    label: "Upload and delete files",
    group: "files",
    risk: "Can delete files",
  },
  "batches:read": {
    label: "List batches and read their results",
    group: "batches",
  },
  "batches:write": {
    label: "Create and cancel batches",
    group: "batches",
    risk: "Can queue spend that runs after the key is revoked",
  },
  "web:search": { label: "Search the web", group: "tools" },
  "web:fetch": { label: "Fetch a web page", group: "tools" },
  "code:execute": {
    label: "Run code in the sandbox",
    group: "tools",
    risk: "Runs arbitrary code",
  },
  "mcp:read": {
    label: "List MCP servers and their tools",
    group: "mcp",
    hint: "Needed to look up a server's id or sign-in link",
  },
  "mcp:execute": {
    label: "Call MCP tools",
    group: "mcp",
    risk: "Acts on the systems those tools reach",
    hint: "Without it, the workspace's MCP servers do nothing for this key",
  },
  "memory:read": {
    label: "Search and recall the workspace's agent memory",
    group: "memory",
    hint: "Without it, agent memory does nothing for this key",
  },
  "memory:write": {
    label: "Save, confirm, flag and forget agent memory entries",
    group: "memory",
    risk: "Changes what every agent in the workspace recalls",
  },
  "telemetry:write": { label: "Send OpenTelemetry data", group: "agents" },
  "agents:write": {
    label: "Coding-agent hooks and model recommendations",
    group: "agents",
  },
  "usage:read": { label: "Read its own budget and usage", group: "usage" },
  "usage:write": { label: "Report usage observed elsewhere", group: "usage" },
  "end-users:read": { label: "List its end users", group: "endUsers" },
  "end-users:write": {
    label: "Create, move and block its end users",
    group: "endUsers",
  },
  "end-users:delete": {
    label: "Delete its end users",
    group: "endUsers",
    risk: "Erases their records for good",
  },
}

/**
 * Each group's heading, in the order the form lays them out, whether only a
 * service key is offered it, and whether it is folded under Advanced: what a
 * new key holds (models, the built-in tools and MCP) stays in view.
 */
const GROUP_HEADINGS: Record<
  ScopeGroupKey,
  { label: string; serviceKeyOnly?: boolean; isAdvanced?: boolean }
> = {
  models: { label: "Models" },
  tools: { label: "Tools" },
  mcp: { label: "MCP" },
  memory: { label: "Agent memory" },
  files: { label: "Files", isAdvanced: true },
  batches: { label: "Batches", isAdvanced: true },
  agents: { label: "Coding agents", isAdvanced: true },
  usage: { label: "Usage", isAdvanced: true },
  endUsers: { label: "End users", serviceKeyOnly: true, isAdvanced: true },
}

export const SCOPES = Object.entries(SCOPE_INFO) as [KeyScope, ScopeInfo][]

/**
 * The groups the form lays out, each with the scopes that named it in the
 * gateway's order. Derived rather than listed, so a scope cannot be left out.
 */
export const GROUPS = (Object.keys(GROUP_HEADINGS) as ScopeGroupKey[]).map(
  (key) => ({
    key,
    ...GROUP_HEADINGS[key],
    scopes: SCOPES.filter(([, info]) => info.group === key).map(
      ([scope]) => scope,
    ),
  }),
)
